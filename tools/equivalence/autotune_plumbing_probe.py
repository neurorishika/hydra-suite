"""Probe the autotuner's APPLY plumbing on a real tracking run (question 3b).

The equivalence matrix answers "do tuned settings perturb tracking output?"
(``AUTOTUNE=1 bash run_matrix.sh``). It deliberately never turns the tuner on,
because a committed seed profile can never hit the tuning cache:
``hydra_code_identity`` hashes all package sources, so the key moves with every
edit, and a seeded-on-disk profile would silently MISS -- leaving a leg that
compares defaults against defaults and prints a meaningless green.

This script answers the *other* question -- does an applied overlay actually
reach the worker -- without that confound. It seeds the profile **in process**:
it wraps ``build_tracking_autotune_request`` so the profile is saved under the
exact key the live run just computed, in the same interpreter, and can therefore
never miss by construction. Then it runs one short real tracking pass in
``automatic`` mode and reports the worker's own overlay line.

It asserts nothing about CSV content -- output equivalence is the matrix's job.
It reports, per run, the resolved ``status``, whether ``effective`` differs from
``requested``, and the request's eligibility fields, which is what distinguishes
"the overlay was applied" from "policy declined to apply anything".

Usage::

    PYTHONPATH=src python tools/equivalence/autotune_plumbing_probe.py \
        --video tools/equivalence/fixtures/clips/fly_obb.mp4 \
        --config tools/equivalence/fixtures/configs/fly_obb.json \
        --outdir /tmp/autotune_plumbing --end-frame 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("equiv.autotune_probe")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--end-frame", type=int, default=60)
    ap.add_argument(
        "--tuned-detection-batch",
        type=int,
        default=4,
        help="detection_batch_size stored in the seeded profile's SELECTED "
        "settings; must differ from the baseline for 'effective != requested' "
        "to be a meaningful assertion.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    video_link = outdir / Path(args.video).name
    video_link.symlink_to(Path(args.video).resolve())

    cfg = json.loads(Path(args.config).read_text())
    cfg["file_path"] = str(video_link)
    cfg["csv_path"] = str(outdir / f"{video_link.stem}_tracking.csv")
    cfg["use_cached_detections"] = False
    cfg["end_frame"] = int(args.end_frame)
    cfg["apply_tuned_inference"] = True
    # A tuner cannot run on a realtime pass (it is declared non-tunable), and
    # this probe is about the apply path, not the calibration path.
    cfg["tracking_workflow_mode"] = "offline"
    cfg["realtime_tracking_mode"] = False
    for key in (
        "video_output_enabled",
        "enable_confidence_density_map",
        "enable_dataset_generation",
        "enable_individual_dataset",
        "enable_individual_image_save",
        "final_media_export_videos_enabled",
    ):
        cfg[key] = False
    cfg_path = outdir / "probe_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    from hydra_suite.core.inference.autotune import integration
    from hydra_suite.core.inference.autotune.models import (
        CandidateEvidence,
        EquivalenceVerdict,
        InferenceTuningProfile,
        ProfileState,
    )
    from hydra_suite.core.inference.autotune.store import InferenceTuningProfileStore

    # integration.resolve_tracking_inference_config builds its default store as
    # `get_data_dir() / "inference_tuning_profiles"`. Redirect ONLY that lookup
    # so the probe's fake profile never lands in the user's real data dir.
    # Setting HYDRA_DATA_DIR instead would also move the models dir and the run
    # would fail for an unrelated reason.
    store_dir = outdir / "profile_store"
    integration.get_data_dir = lambda: outdir  # type: ignore[assignment]
    observed: dict[str, object] = {}
    real_builder = integration.build_tracking_autotune_request

    def seeding_builder(*a, **kw):  # type: ignore[no-untyped-def]
        request = real_builder(*a, **kw)
        # Everything below uses the request the LIVE run just built, so the
        # seeded key is the live key by construction -- no digest drift.
        observed["key_digest"] = request.key.digest
        observed["eligible"] = request.eligible
        observed["allow_cached_reuse"] = request.allow_cached_reuse
        observed["eligibility_reason"] = request.eligibility_reason
        observed["baseline"] = request.baseline.to_dict()
        selected = replace(
            request.baseline, detection_batch_size=int(args.tuned_detection_batch)
        )
        observed["selected"] = selected.to_dict()
        profile = InferenceTuningProfile(
            request.key.digest[:24],
            request.key,
            request.baseline,
            request.baseline,
            selected,
            selected,
            (
                CandidateEvidence(
                    selected,
                    (120.0,) * 5,
                    stage_seconds_samples=(0.5,) * 5,
                    warmup_calls=3,
                    warmup_frames=8,
                    equivalence=EquivalenceVerdict(True),
                ),
            ),
            ProfileState.VALIDATED,
            "seeded-by-probe",
        )
        InferenceTuningProfileStore(store_dir).save(profile)
        observed["seeded"] = True
        return request

    integration.build_tracking_autotune_request = seeding_builder  # type: ignore[assignment]

    overlay_lines: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if "Inference throughput autotuner: status=" in message:
                overlay_lines.append(message)

    handler = _Capture()
    logging.getLogger("hydra_suite.core.tracking.worker").addHandler(handler)

    from hydra_suite.trackerkit.cli import run_tracking_cli

    try:
        rc = run_tracking_cli([str(video_link)], config_path=str(cfg_path))
    finally:
        integration.build_tracking_autotune_request = real_builder
        logging.getLogger("hydra_suite.core.tracking.worker").removeHandler(handler)

    report = {
        "exit_code": rc,
        "request": {k: v for k, v in observed.items() if k != "seeded"},
        "profile_seeded": bool(observed.get("seeded")),
        "overlay_lines": overlay_lines,
    }
    print(json.dumps(report, indent=2, default=str))
    (outdir / "probe_report.json").write_text(json.dumps(report, indent=2, default=str))

    if not overlay_lines:
        log.error("no autotuner overlay line was emitted -- plumbing NOT exercised")
        return 2
    line = overlay_lines[-1]
    applied = "status=cache_hit" in line and "effective=" in line
    log.info("cache_hit=%s", applied)
    return 0


if __name__ == "__main__":
    sys.exit(main())
