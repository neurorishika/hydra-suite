# External ViTPose checkpoints — out-of-the-box probe findings

Date: 2026-08-03. Probe tool: `tools/vitpose/external_ckpt/`.

## What was tested

Two checkpoints trained by a collaborator (`tywei08/ViTPose_checkpoints`, private),
run out of the box on individual animals cropped from our own DEMO footage.

| checkpoint | keypoints | sha256 |
|---|---|---|
| `ViTPose_base_ant9kp_256x256.pth` | 9 | `30ff9a51bd1293c4160f0ec1d8fe5b6596556ff0f50e810c93c23680e5676cb8` |
| `ViTPose_base_fly29kp_ImgAug_256x256.pth` | 29 | `8b078bb6cd8d7d044b1f518499b0c8b2557de87b6072dd3432326781d922dc57` |

Both are ViT-base / 256x256 / `TopdownHeatmapSimpleHead`, mmpose 0.x configs.
Crops came from the completed tracking CSVs in `DEMO 3` (ant) and `DEMO 4`
(melanogaster) — no detector was re-run. 12 samples per species per crop mode,
spread evenly across the tracked frame range.

## Architecture compatibility: exact

The repo's pure-torch `ViT` + `ClassicHead` is a byte-compatible reimplementation
of what these checkpoints contain. **Both load with `strict=True`**, no rename map,
no missing or unexpected keys:

- `backbone.pos_embed` is `(1, 257, 768)` — matches `ViT(img_size_hw=(256, 256))`
  exactly (16x16 patch grid + the MAE cls slot).
- `keypoint_head.final_layer.weight` is `(9|29, 256, 1, 1)` — matches `ClassicHead`.
- 163 state keys, prefixes `backbone.*` / `keypoint_head.*`.

No change under `src/` was needed. `ViT` already accepts `img_size_hw`; only the
pre/post-processing constants are baked to 192x256, and this tool supplies its own.

### Loading gotcha

These are full mmpose training blobs — top level is `['meta', 'state_dict', 'optimizer']`
(hence 1.0 GB each), and `meta` holds numpy scalars. `torch.load(weights_only=True)`
therefore raises `UnpicklingError: Unsupported global: numpy.core.multiarray.scalar`.

`model.py` fixes this with a precise `add_safe_globals` allowlist of numpy
primitives only — **not** `weights_only=False`, which would unpickle arbitrary code
from a downloaded file. Note that numpy 2.x renamed `numpy.core` to `numpy._core`,
so both spellings are registered: the downloaded checkpoints carry the old name,
while a checkpoint written by this environment carries the new one.

## Result 1 — the ant checkpoint transfers. Usable for the body axis.

At `--scale 1.5` the body-axis chain `Head_T -> Centroid -> Abd_T -> Abd_B` sits on
the ant's midline in roughly 10 of 12 sampled crops, with the head end correctly
identified rather than flipped.

Antennae are the weak point. `A_L_T`/`A_R_T` (the tips) frequently splay into
background instead of following the real antennae; the right antenna overshoots more
often than the left. The confidence table agrees: `Abd_T` (median 0.71) and `Head_B`
(0.77) are the softest rows, the rest sit near 0.85-0.93.

Occasional failure: one crop showed an apparent 180-degree head/tail inversion,
which is consistent with this pipeline's known bistable head/tail ambiguity rather
than a model defect.

## Result 2 — the fly checkpoint does not usefully transfer.

Body-axis keypoints (`headTop`, `thoraxCenter`, `abdomenTop`, `abdomenCenter`,
`genitalia`) become credible at tight crops (see below), but **all 18 leg keypoints
are unreliable in every configuration tested** — they scatter onto background rather
than onto legs, producing a starburst radiating from `thoraxCenter`.

Confidences are high and misleading (medians 0.6-0.95 including the leg rows). That
is the expected signature of a heatmap model falling back on a learned mean pose:
peaks stay sharp while localization is meaningless. Do not read these confidences as
evidence the predictions are good.

Likely cause is a domain gap, not a bug: our DEMO footage is grayscale and backlit
with frequently touching/overlapping flies, whereas the checkpoint was trained on
`fly-29kp-2cls-ImgAug`.

## Result 3 — crop framing matters more than orientation

**Orientation does not matter.** Despite `rot_factor=0` in both training configs
(no rotation augmentation), axis-aligned crops performed at least as well as
heading-rotated ones for the ant, and rotation did not rescue the fly. These
checkpoints are not orientation-locked, so a deployment does not need to canonicalise
heading first. This was the main open question going in, and the answer is negative.

**Scale matters a lot.** The initial `--scale 2.0` guess was too generous:

- Ant: at 2.0 the crop pulls in neighbouring ants and some skeletons span two
  animals. At **1.5** this largely disappears and results improve markedly.
- Fly: at 2.0 the predictions are near-useless. At **1.2** the body axis becomes
  credible. Legs remain wrong at every scale tested (1.2, 1.5, 2.0, 3.0).

The CLI default remains `--scale 2.0` (as specified in the plan); **1.5 for ant and
1.2 for fly are the empirically better settings** and should be passed explicitly.

## Reproducing

```bash
conda activate hydra-mps
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=$PWD/src
python -m tools.vitpose.external_ckpt.cli --species ant \
  --ckpt <dir>/ViTPose_base_ant9kp_256x256.pth --scale 1.5 --out /tmp/vitpose_probe
python -m tools.vitpose.external_ckpt.cli --species fly \
  --ckpt <dir>/ViTPose_base_fly29kp_ImgAug_256x256.pth --scale 1.2 --out /tmp/vitpose_probe
```

Writes `<species>_{axis,rot}.png` contact sheets and `<species>_{axis,rot}_confidence.txt`.

## Recommendation

**Do not promote either checkpoint to first-class support yet.** Promotion would mean
parameterising `IMAGE_SIZE_WH`/`HEATMAP_SIZE_WH` in `src/`, which carries
byte-identical-parity exposure; neither result yet justifies that cost.

- The **ant** checkpoint is worth a quantitative follow-up: build a keypoint
  correspondence between its 9-point schema and ours, then score PCK/OKS on labelled
  ant data. If the body axis holds up numerically it is a credible warm-start for
  fine-tuning, and fine-tuning is the natural way to fix the antennae.
- The **fly** checkpoint should be treated as a fine-tuning initialisation at best,
  not an out-of-the-box predictor. The leg keypoints — which are most of its value
  over our existing 8-point schemas — do not survive the domain gap.
