"""``detectkit escalate``: headless SAM2 / SAM3 escalation.

Runs the same jobs the GUI's "Escalate to segment (SAM2)" and "Semantic
escalation (SAM3)" actions run, against a saved DetectKit project, and leaves
the result STAGED for review exactly as the GUI does -- open the project in
DetectKit to accept or reject the staged frames.

    detectkit escalate sam2 --project DIR [--source NAME ...] [--device mps]
    detectkit escalate sam3 --project DIR --prompt "ant" [--class-name ant]

Runs in this process: unlike the GUI it does not use the DetectKit sidecar's
memory containment, so size --device and --max-instances for the host.

Works on every device (Auto picks CUDA, then Apple MPS, then CPU). SAM3's
stock weights are licence-gated: accept the licence at
https://huggingface.co/facebook/sam3 and run ``hf auth login`` once per
machine before the first SAM3 run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

DEVICES = ("auto", "cuda", "mps", "cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="detectkit escalate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="model", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--project", required=True, help="DetectKit project directory")
        sp.add_argument(
            "--source",
            action="append",
            default=[],
            help="source name to escalate (repeatable; default: every eligible source)",
        )
        sp.add_argument("--device", choices=DEVICES, default="auto")
        sp.add_argument(
            "--overwrite",
            action="store_true",
            help="replace an existing pending (unreviewed) escalation",
        )

    from hydra_suite.core.inference.sam2.checkpoints import (
        DEFAULT_VARIANT as SAM2_DEFAULT,
    )
    from hydra_suite.core.inference.sam2.checkpoints import (
        available_variants as sam2_variants,
    )

    s2 = sub.add_parser("sam2", help="box/OBB labels -> SAM2 polygon masks")
    common(s2)
    s2.add_argument("--variant", choices=sam2_variants(), default=SAM2_DEFAULT)
    s2.add_argument(
        "--tile-fraction",
        type=float,
        default=None,
        help="segment each box in a tile of body size / this fraction (0 = full "
        "frame). Default: what the DetectKit dialog last used for this variant, "
        "else full frame.",
    )
    s2.add_argument(
        "--reference-body-px",
        type=float,
        default=None,
        help="typical longest animal side in px (default: the dialog's value, then "
        "the project's sliced-training reference, then the label median)",
    )

    s3 = sub.add_parser("sam3", help="text prompt -> SAM3 instance masks")
    common(s3)
    s3.add_argument("--prompt", required=True, help='noun phrase, e.g. "ant"')
    s3.add_argument(
        "--class-name",
        default="",
        help="project class the staged instances are labelled as (must be one of the "
        "project's classes; default: the only class, or the prompt if it is one)",
    )
    s3.add_argument(
        "--variant",
        default="sam3",
        help="stock variant or a published finetuned model key",
    )
    s3.add_argument("--confidence", type=float, default=0.35)
    s3.add_argument("--max-instances", type=int, default=0, help="0 = unlimited")
    s3.add_argument(
        "--tile-fraction",
        type=float,
        default=None,
        help="tile = body size / this fraction (0 = full frame). Default: what "
        "the DetectKit dialog would open with (saved -> calibrated -> the "
        "model's stamp -> the seed at the project's body size).",
    )
    s3.add_argument(
        "--reference-body-px",
        type=float,
        default=None,
        help="typical longest animal side in px (default: the dialog's value, then "
        "the project's sliced-training reference, then the label median)",
    )
    return p


def _progress(pct: int, msg: str) -> None:
    print(f"[{int(pct):3d}%] {msg}", flush=True)


def _open(project_dir: str):
    from hydra_suite.detectkit.gui.project import open_project

    path = Path(project_dir).expanduser().resolve()
    project = open_project(path)
    if project is None:
        raise SystemExit(f"error: could not open DetectKit project at {path}")
    return project


def _selected(project, names: List[str]):
    sources = list(project.sources)
    if not names:
        return sources
    by_name = {s.name: s for s in sources}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise SystemExit(
            f"error: unknown source(s) {missing}; project has {sorted(by_name)}"
        )
    return [by_name[n] for n in names]


def _sam2_tiling(project, args: argparse.Namespace) -> dict:
    """The SAM2 tiling for this run: explicit flags, else the dialog's default.

    The default comes from ``default_geometry_tiling``, the same function the
    dialog opens with, so a headless run stages what the GUI would.
    """
    from hydra_suite.detectkit.jobs.calibration_frames import (
        has_polygon_frames,
        quick_median_body_px,
    )
    from hydra_suite.detectkit.jobs.sam2_escalation import default_geometry_tiling

    # No flags: exactly what the dialog would open with for this variant.
    base = default_geometry_tiling(project, args.variant)
    fraction = (
        base["tile_fraction"] if args.tile_fraction is None else args.tile_fraction
    )
    fraction = fraction if fraction and fraction > 0 else None
    body = (
        base["reference_body_px"]
        if args.reference_body_px is None
        else float(args.reference_body_px)
    )
    if body <= 0 and args.tile_fraction is not None and fraction is not None:
        # An explicit --tile-fraction with no known body size: resolve one
        # the way the dialog prefills it, rather than silently not tiling.
        slice_settings = getattr(project, "slice_settings", None)
        body = float(getattr(slice_settings, "reference_body_px", 0.0) or 0.0)
        if body <= 0:
            sources = list(project.sources)
            body = quick_median_body_px(
                [s for s in sources if has_polygon_frames(s)] or sources
            )
    return {"reference_body_px": float(body or 0.0), "tile_fraction": fraction}


def _sam3_tiling(project, args: argparse.Namespace) -> dict:
    """The SAM3 tiling for this run: explicit flags, else the dialog's default.

    The default comes from ``default_semantic_tiling``, the same function the
    dialog opens with, so a headless run stages what the GUI would (F3).
    """
    from hydra_suite.detectkit.jobs.calibration_frames import (
        has_polygon_frames,
        quick_median_body_px,
    )
    from hydra_suite.detectkit.jobs.semantic_escalation import default_semantic_tiling

    if args.reference_body_px is not None and args.tile_fraction is None:
        # M2: an explicit body with no explicit fraction is the dialog's body
        # field filled in -- resolve the opening state AT that body, so a
        # stock variant tiles at the seed instead of falling to full frame
        # because the project's own body chain was empty.
        base = default_semantic_tiling(
            project, args.variant, body_chain_px=float(args.reference_body_px)
        )
    else:
        base = default_semantic_tiling(project, args.variant)
    fraction = (
        base["tile_fraction"] if args.tile_fraction is None else args.tile_fraction
    )
    fraction = fraction if fraction and fraction > 0 else None
    body = (
        float(base["reference_body_px"] or 0.0)
        if args.reference_body_px is None
        else float(args.reference_body_px)
    )
    if body <= 0 and args.tile_fraction is not None and fraction is not None:
        # An explicit --tile-fraction with no known body size: resolve one the
        # way the dialog prefills it (same chain as _sam2_tiling).
        slice_settings = getattr(project, "slice_settings", None)
        body = float(getattr(slice_settings, "reference_body_px", 0.0) or 0.0)
        if body <= 0:
            sources = list(project.sources)
            body = quick_median_body_px(
                [s for s in sources if has_polygon_frames(s)] or sources
            )
    origin = "flags" if args.tile_fraction is not None else base["origin"]
    return {
        "reference_body_px": float(body or 0.0),
        "tile_fraction": fraction,
        "overlap": float(base["overlap"]),
        "merge_iou": float(base["merge_iou"]),
        "area_min_px2": float(base["area_min_px2"]),
        "area_max_px2": float(base["area_max_px2"]),
        "origin": origin,
    }


def run_sam2(args: argparse.Namespace) -> int:
    from hydra_suite.core.inference.sam2.executor import Sam2SegmentExecutor
    from hydra_suite.core.inference.torch_device import resolve_torch_device
    from hydra_suite.detectkit.gui.project import save_project
    from hydra_suite.detectkit.jobs.sam2_escalation import (
        EscalationRequest,
        run_escalation,
    )

    project = _open(args.project)
    sources = [s for s in _selected(project, args.source) if s.level != "polygon"]
    if not sources:
        print("Nothing to escalate: every selected source is already a polygon source.")
        return 0
    device = resolve_torch_device(args.device)
    tiling = _sam2_tiling(project, args)
    print(f"SAM2 {args.variant} on {device}: {', '.join(s.name for s in sources)}")
    from hydra_suite.core.inference.semantic.tiling import resolve_tile_px

    if resolve_tile_px(tiling["reference_body_px"], tiling["tile_fraction"]):
        print(
            f"Tiling: fraction {tiling['tile_fraction']:g} of a "
            f"{tiling['reference_body_px']:.0f} px body"
        )
    executor = Sam2SegmentExecutor.from_variant(args.variant, device=device)
    request = EscalationRequest(
        project=project,
        source_names=[s.name for s in sources],
        source_paths=[s.path for s in sources],
        variant=args.variant,
        overwrite=args.overwrite,
        device=args.device,
        **tiling,
    )
    result = run_escalation(
        request,
        executor,
        overwrite=args.overwrite,
        progress=_progress,
        on_mutated=lambda: save_project(project),
    )
    save_project(project)
    for name, reason in result.skipped:
        print(f"skipped {name}: {reason}")
    print(
        f"Done: {len(result.staged)} source(s) staged, {result.primed} primed, "
        f"{result.fell_back} fell back; {result.tiled_frames} frame(s) tiled, "
        f"{result.untiled_frames} could not be, {result.seam_fallbacks} seam "
        "fallback(s). Open the project in DetectKit to review."
    )
    return 0


def _resolve_class_name(project, requested: str, prompt: str) -> str:
    """The project class staged instances are written as -- never free text.

    The prompt is not the class: an instance labelled with a name outside
    ``project.class_names`` is silently dropped by the overlay and the training
    dataset builder. Mirror the GUI, which only offers the project's classes.
    """
    classes = [str(c) for c in (getattr(project, "class_names", None) or []) if str(c)]
    if not classes:
        return requested  # no class scheme: the job falls back to the prompt
    chosen = requested or (prompt if prompt in classes else "")
    if not chosen and len(classes) == 1:
        chosen = classes[0]
    if chosen not in classes:
        raise SystemExit(
            f"error: --class-name must be one of the project's classes {classes} "
            f"(got {chosen or 'nothing'!r}; the prompt {prompt!r} is not a class)."
        )
    return chosen


def run_sam3(args: argparse.Namespace) -> int:
    from hydra_suite.core.inference.semantic.checkpoints import (
        Sam3DownloadNotAuthorized,
        probe_dependencies,
    )
    from hydra_suite.detectkit.sidecars.operations import (
        run_semantic_escalation_sidecar,
    )

    deps = probe_dependencies()
    if not deps.usable:
        print(f"error: SAM3 is not usable here: {deps.reason}", file=sys.stderr)
        return 2
    project = _open(args.project)
    sources = _selected(project, args.source)
    class_name = _resolve_class_name(project, args.class_name, args.prompt)
    tiling = _sam3_tiling(project, args)
    origin = tiling.pop("origin")
    payload = {
        "project_dir": str(Path(project.project_dir).expanduser().resolve()),
        "source_names": [s.name for s in sources],
        "source_paths": [s.path for s in sources],
        "variant": args.variant,
        "prompt": args.prompt,
        "class_name": class_name,
        "device": args.device,
        "params": {
            "device": args.device,
            "confidence": float(args.confidence),
            "max_instances": int(args.max_instances),
            "overwrite": bool(args.overwrite),
            # SemanticEscalationRequest fields: without these the request's
            # body default (0) silently ran every headless job full frame.
            **tiling,
        },
    }
    print(f"SAM3 {args.variant!r} prompt={args.prompt!r} device={args.device}")
    from hydra_suite.core.inference.semantic.tiling import resolve_tile_px

    if resolve_tile_px(tiling["reference_body_px"], tiling["tile_fraction"]):
        print(
            f"Tiling: fraction {tiling['tile_fraction']:g} of a "
            f"{tiling['reference_body_px']:.0f} px body ({origin})"
        )
    else:
        print("Tiling: full frame")
    try:
        out = run_semantic_escalation_sidecar(payload, _progress)
    except Sam3DownloadNotAuthorized as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    result = out.get("semantic_result", {})
    print(f"Done: {result}. Open the project in DetectKit to review.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_sam2(args) if args.model == "sam2" else run_sam3(args)


if __name__ == "__main__":
    sys.exit(main())
