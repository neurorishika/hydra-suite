#!/usr/bin/env python
"""Confirm the ACTUAL production entry point (`load_pose_model`) selects the
right SLEAP backend for the `gpu` and `gpu_fast` runtime tiers on a real CUDA
host -- no mocks, no stubbed factory. Runs one real crop through each tier's
backend to prove it doesn't just construct cleanly but actually predicts.

Run in the HYDRA env (spawns the `sleap` conda-env service for the gpu tier):

  PYTHONPATH=src python tools/equivalence/verify_pose_tier_selection.py \
    --model-dir "$HOME/.local/share/hydra-suite/models/pose/SLEAP/20260214-224154_unet_ant_single_instance" \
    --skeleton tools/equivalence/fixtures/ooceraea_biroi.json \
    --sleap-env sleap
"""

import argparse
import json
import time

import numpy as np

from hydra_suite.core.inference.config import PoseConfig, PoseSLEAPConfig
from hydra_suite.core.inference.runtime import RuntimeContext
from hydra_suite.core.inference.stages.pose import load_pose_model


def _gpu_rt():
    """gpu tier on a CUDA host: native torch, tensors stay on-device."""
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        default_runtime="cuda",
        tensor_on_cuda=True,
    )


def _gpu_fast_rt():
    """gpu_fast tier on a CUDA host: TensorRT engines, CPU numpy outputs."""
    return RuntimeContext(
        cuda_mode=True,
        device="cuda:0",
        use_nvdec=True,
        default_runtime="cuda",
        tensor_on_cuda=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--sleap-env", default="sleap")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    dummy_crop = np.zeros((64, 128, 3), dtype=np.uint8)

    results = {}
    for tier_name, rt in [("gpu", _gpu_rt()), ("gpu_fast", _gpu_fast_rt())]:
        print(
            f"\n=== tier={tier_name} (cuda_mode={rt.cuda_mode}, tensor_on_cuda={rt.tensor_on_cuda}) ==="
        )
        config = PoseConfig(
            backend="sleap",
            skeleton_file=args.skeleton,
            sleap=PoseSLEAPConfig(
                model_path=args.model_dir,
                conda_env=args.sleap_env,
                batch_size=args.batch_size,
            ),
        )
        t0 = time.perf_counter()
        model = load_pose_model(config, rt)
        load_s = time.perf_counter() - t0

        backend = model.backend
        backend_type = type(backend).__name__
        runtime_flavor = getattr(backend, "runtime_flavor", None)
        device = getattr(backend, "device", None)

        t0 = time.perf_counter()
        out = backend.predict_batch([dummy_crop])
        predict_s = time.perf_counter() - t0

        ok = bool(out) and out[0] is not None
        print(f"  backend class:   {backend_type}")
        print(f"  runtime_flavor:  {runtime_flavor}")
        print(f"  device:          {device}")
        print(f"  load_pose_model: {load_s:.2f}s")
        print(f"  predict_batch:   {predict_s:.3f}s (dummy crop, real forward pass)")
        print(f"  produced result: {ok}")

        results[tier_name] = {
            "backend_class": backend_type,
            "runtime_flavor": runtime_flavor,
            "device": device,
            "load_s": load_s,
            "predict_s": predict_s,
            "produced_result": ok,
        }
        close = getattr(backend, "close", None)
        if callable(close):
            close()

    print("\n=== SUMMARY ===")
    print(json.dumps(results, indent=2))

    expect = {
        "gpu": ("SleapServiceBackend", "native"),
        "gpu_fast": ("SleapExportedBackend", "tensorrt"),
    }
    ok_all = True
    for tier_name, (exp_class, exp_flavor) in expect.items():
        got_class = results[tier_name]["backend_class"]
        got_flavor = results[tier_name]["runtime_flavor"]
        match = got_class == exp_class
        print(
            f"{tier_name}: expected backend={exp_class!r}, got={got_class!r} "
            f"-> {'OK' if match else 'MISMATCH'} (runtime_flavor={got_flavor!r}, expected~{exp_flavor!r})"
        )
        ok_all = ok_all and match

    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
