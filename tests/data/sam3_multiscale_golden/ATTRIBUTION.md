# SAM3 single-scale dataset golden

`single_scale_tree.json` is a characterization golden: the sorted relative path
-> SHA-256 tree hash of the dataset produced by
`hydra_suite.training.sam3_lora.dataset_build.build_sam3_coco_dataset` at the
**contract defaults** (`geometry_mode="auto_object"`,
`object_tile_fraction=0.055`, `tile_overlap=0.25`, `keep_empty_tiles=True`),
`seed=42`, default `SplitConfig()`, over the deterministic synthetic corpus
built by `write_corpus()` in `tests/test_sam3_multiscale_gate.py`
(`CORPUS_SEED=20260906`, 4 x 2048px PNG frames, 3 x 40px square bodies each).

It exists to make "multi-scale changed X" a measured claim: the multi-scale
port (plan `docs/superpowers/plans/2026-09-06-multiscale-sam3-training.md`)
specifies that `object_tile_fractions=()` and `full_frame_mix=False` fork to
today's exact code path, and this hash is what proves it.

- Produced at commit: `0d4d4cae` (branch `feat/multiscale-sam3`), 2026-09-06.
- 67 entries: `build_manifest.json`, `train/` + `valid/`
  `_annotations.coco.json`, and every tile JPEG.
- Normalisation: JSON files are re-serialised with sorted keys before hashing;
  `created_at` and the absolute `source` path are dropped from
  `build_manifest.json`. Nothing else is normalised -- the build was verified
  byte-identical across two runs from different source/output directories.
- Environment (JPEG bytes are encoder-pinned): OpenCV 4.13.0,
  macOS-26.5.1-arm64 (Apple silicon), Python 3.13.12, conda env `hydra-mps`.
  On a different libjpeg build the tile-image hashes may differ while paths,
  COCO JSON and manifest hashes stay identical; that is an encoder difference,
  not a builder behaviour change.

Regenerate deliberately (and review the diff) with:

    HYDRA_UPDATE_SAM3_GOLDEN=1 PYTHONPATH=$PWD/src \
      python -m pytest tests/test_sam3_multiscale_gate.py
