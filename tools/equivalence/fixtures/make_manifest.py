"""Build the release manifest + the models tarball for the equivalence fixtures.

Run after generate_clips.py, on a machine that has the required model files in
its hydra-suite models dir. Produces, under fixtures/staging/:

  - models.tar.gz   (the exact model files the clip configs reference, packed
                     with paths relative to the models dir; a SLEAP model is a
                     directory, hence a tarball rather than per-file assets)

and writes fixtures/manifest.json with sha256 + sizes for every release asset
(the clips and models.tar.gz). fetch_fixtures.sh consumes this manifest.

The model list is DERIVED from the portable configs (every relative model path
that resolves under the models dir), so adding a clip in generate_clips.py is
enough — no second list to maintain. A multihead classifier descriptor
(*.multihead.json) pulls in its sibling factor .pth weights automatically.

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
CONFIGS_DIR = HERE / "configs"
STAGING = HERE / "staging"

RELEASE_TAG = "equiv-fixtures-v2"

# Clips and the pose skeleton each needs (None when pose is off). The model set
# is derived from each clip's config, not listed here.
CLIPS = [
    {"name": "emi_obb_identity.mp4", "skeleton": None},
    {"name": "ant_pose_headtail.mp4", "skeleton": "ooceraea_biroi.json"},
    {"name": "ant_obb_sleap.mp4", "skeleton": "ooceraea_biroi.json"},
    {"name": "worm_bgsub.mp4", "skeleton": None},
    {"name": "ant_cnn_identity.mp4", "skeleton": "ooceraea_biroi.json"},
    {"name": "fly_obb.mp4", "skeleton": None},
]

# Configs that reuse an existing clip above (no separate .mp4 asset) but
# reference model files that still need to ship in models.tar.gz.
EXTRA_MODEL_CONFIGS = ["ant_obb_sequential.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _walk_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)


def collect_models(models_dir: Path) -> list[str]:
    """Every relative model path referenced by a portable config that exists
    under the models dir, plus the factor weights of any multihead descriptor."""
    rels: set[str] = set()
    cfg_names = [Path(c["name"]).stem + ".json" for c in CLIPS] + EXTRA_MODEL_CONFIGS
    for cfg_name in cfg_names:
        cfg_path = CONFIGS_DIR / cfg_name
        cfg = json.loads(cfg_path.read_text())
        for s in _walk_strings(cfg):
            # Real model paths are relative and live under a category subdir
            # (obb/…, pose/SLEAP/…, classification/…). The "/" guard rejects bare
            # tokens like task_family="obb" that would otherwise match the whole
            # obb/ directory.
            if not s or s.startswith("/") or "/" not in s:
                continue
            target = models_dir / s
            if not target.exists():
                continue
            rels.add(s)
            # A multihead classifier descriptor references sibling .pth weights.
            if s.endswith(".multihead.json"):
                desc = json.loads(target.read_text())
                parent = Path(s).parent
                for fm in desc.get("factor_models", []):
                    w = fm.get("path")
                    if w and (models_dir / parent / w).exists():
                        rels.add(str(parent / w))
    return sorted(rels)


def build_models_tar(models_dir: Path, models: list[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for rel in models:
            src = models_dir / rel
            if not src.exists():
                raise FileNotFoundError(f"missing model: {src}")
            tar.add(src, arcname=rel)


def main() -> None:
    models_dir = Path(get_models_dir())
    models = collect_models(models_dir)
    tar_path = STAGING / "models.tar.gz"
    print(f"packing {len(models)} models from {models_dir} -> {tar_path}")
    for m in models:
        print(f"  + {m}")
    build_models_tar(models_dir, models, tar_path)

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
        "models_contained": models,
    }
    for c in CLIPS:
        clip = CLIPS_DIR / c["name"]
        if not clip.exists():
            raise FileNotFoundError(f"missing clip (run generate_clips.py): {clip}")
        manifest["clips"].append(
            {
                "name": c["name"],
                "sha256": sha256(clip),
                "bytes": clip.stat().st_size,
                "config": f"configs/{Path(c['name']).stem}.json",
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
