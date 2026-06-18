"""Build the release manifest + the models tarball for the equivalence fixtures.

Run after generate_clips.py, on a machine that has the required model files in
its hydra-suite models dir. Produces, under fixtures/staging/:

  - models.tar.gz   (the exact model files the clip configs reference, packed
                     with paths relative to the models dir; the SLEAP model is a
                     directory, hence a tarball rather than per-file assets)

and writes fixtures/manifest.json with sha256 + sizes for every release asset
(the two clips and models.tar.gz). fetch_fixtures.sh consumes this manifest.

Upload the assets named in the manifest to a GitHub Release tagged RELEASE_TAG.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from hydra_suite.paths import get_models_dir

HERE = Path(__file__).resolve().parent
CLIPS_DIR = HERE / "clips"
STAGING = HERE / "staging"

RELEASE_TAG = "equiv-fixtures-v1"

# Clips and the models each needs. Model paths are relative to the models dir.
CLIPS = [
    {
        "name": "emi_obb_identity.mp4",
        "config": "configs/emi_obb_identity.json",
        "skeleton": None,
    },
    {
        "name": "ant_pose_headtail.mp4",
        "config": "configs/ant_pose_headtail.json",
        "skeleton": "ooceraea_biroi.json",
    },
]
MODELS = [
    "obb/20260416-165920_26s_obiroi_train36.pt",
    "obb/20260214-210051_26x_ant_train22.pt",
    "classification/orientation/20260429-104937_efficientnet_b0_obiroi_train1.pth",
    "pose/SLEAP/20260214-224154_unet_ant_single_instance",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_models_tar(models_dir: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for rel in MODELS:
            src = models_dir / rel
            if not src.exists():
                raise FileNotFoundError(f"missing model: {src}")
            tar.add(src, arcname=rel)


def main() -> None:
    models_dir = Path(get_models_dir())
    tar_path = STAGING / "models.tar.gz"
    print(f"packing {len(MODELS)} models from {models_dir} -> {tar_path}")
    build_models_tar(models_dir, tar_path)

    manifest = {
        "release_tag": RELEASE_TAG,
        "repo": "neurorishika/hydra-suite",
        "clips": [],
        "models_archive": {
            "name": "models.tar.gz",
            "sha256": sha256(tar_path),
            "bytes": tar_path.stat().st_size,
            "extract_to": "models_dir",
        },
        "models_contained": MODELS,
    }
    for c in CLIPS:
        clip = CLIPS_DIR / c["name"]
        manifest["clips"].append(
            {
                "name": c["name"],
                "sha256": sha256(clip),
                "bytes": clip.stat().st_size,
                "config": c["config"],
                "skeleton": c["skeleton"],
            }
        )

    out = HERE / "manifest.json"
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {out}")
    print(f"models.tar.gz = {manifest['models_archive']['bytes'] / 1e6:.1f} MB")
    for c in manifest["clips"]:
        print(f"  {c['name']} = {c['bytes'] / 1e6:.1f} MB")
    print(f"\nNext: create release '{RELEASE_TAG}' and upload these assets:")
    print(f"  {tar_path}")
    for c in CLIPS:
        print(f"  {CLIPS_DIR / c['name']}")


if __name__ == "__main__":
    main()
