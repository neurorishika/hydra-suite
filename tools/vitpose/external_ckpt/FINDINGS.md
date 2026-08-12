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

Occasional failure: one crop showed an apparent 180-degree head/tail inversion --
the skeleton's head keypoints landed on the abdomen. This is a model error, not
an artifact of this pipeline's tracker: `Theta` only ever rotates the crop before
it is handed to the model (in `rot` mode) and is not used at all in `axis` mode,
so it cannot change which end of the animal the head keypoints land on in the
pixels. A skeleton with the head on the abdomen is wrong in the image itself,
in both crop modes.

## Result 2 — the fly checkpoint does not usefully transfer.

Body-axis keypoints (`headTop`, `thoraxCenter`, `abdomenTop`, `abdomenCenter`,
`genitalia`) become credible at tight crops (see below), but **all 18 leg keypoints
are unreliable in every configuration tested** — they scatter onto background rather
than onto legs, producing a starburst radiating from `thoraxCenter`.

Confidences are high and misleading (medians 0.6-0.95 including the leg rows). That
is the expected signature of a heatmap model falling back on a learned mean pose:
peaks stay sharp while localization is meaningless. Do not read these confidences as
evidence the predictions are good. (`conf` here is the raw unnormalised heatmap peak
value from `decode_default` -- see note below; it is not a calibrated probability.)

Likely cause is a domain gap, not a bug: our DEMO footage is grayscale and backlit
with frequently touching/overlapping flies, whereas the checkpoint was trained on
`fly-29kp-2cls-ImgAug`.

## Result 3 — crop framing matters more than orientation

**No orientation dependence was visible at n=12 -- treat this as a working hypothesis,
not an established result.** Despite `rot_factor=0` in both training configs (no
rotation augmentation), axis-aligned crops performed at least as well as
heading-rotated ones for the ant. For the fly, rotation did not rescue the body-axis
or leg keypoints either, but that arm is weak evidence: nothing rescued the fly
(scale changes didn't, rotation didn't), so the fly comparison mostly says the
checkpoint is broken on this data regardless of orientation, not that orientation
is irrelevant to it. This was the main open question going in; at n=12 the answer
leans negative for the ant, but it should be confirmed quantitatively (e.g. PCK/OKS
vs. crop angle over a larger sample) before a deployment skips heading
canonicalisation on the strength of this probe alone.

**Scale matters a lot.** The initial `--scale 2.0` guess was too generous:

- Ant: at 2.0 the crop pulls in neighbouring ants and some skeletons span two
  animals. At **1.5** this largely disappears and results improve markedly.
- Fly: at 2.0 the predictions are near-useless. At **1.2** the body axis becomes
  credible. Legs remain wrong at every scale tested (1.2, 1.5, 2.0, 3.0).

The CLI default remains `--scale 2.0` (as specified in the plan); **1.5 for ant and
1.2 for fly are the empirically better settings** and should be passed explicitly.

## Per-keypoint confidence (raw, --scale 2.0)

These tables are the verbatim output of `confidence_table()` from the `--scale 2.0`
run -- the run the prose above quotes figures from. Note this predates the scale
sweep in Result 3: the recommended scales (1.5 for ant, 1.2 for fly) were found
afterward and are not reflected here.

`conf` is the raw, unnormalised heatmap peak value returned by `decode_default`
(the value at the argmax pixel of the model's output heatmap). It is **not** a
calibrated probability -- a value of 0.9 does not mean "90% confident" in any
statistical sense, and values are only meaningfully comparable within the same
model/checkpoint, not across models or against some universal threshold.

### `ant_axis_confidence.txt`

```
keypoint                 median      min      max
A_R_T                     0.902    0.061    0.987
A_L_T                     0.920    0.056    1.033
A_R_M                     0.850    0.432    1.052
A_L_M                     0.933    0.560    0.990
Head_T                    0.909    0.771    1.020
Centroid                  0.848    0.686    1.001
Abd_T                     0.714    0.606    0.868
Abd_B                     0.819    0.769    0.978
Head_B                    0.766    0.370    0.902
```

### `ant_rot_confidence.txt`

```
keypoint                 median      min      max
A_R_T                     0.937    0.059    1.035
A_L_T                     0.966    0.111    1.041
A_R_M                     0.847    0.182    0.974
A_L_M                     0.791    0.449    0.966
Head_T                    0.929    0.778    1.031
Centroid                  0.774    0.542    0.940
Abd_T                     0.683    0.455    0.837
Abd_B                     0.794    0.699    0.891
Head_B                    0.695    0.315    0.844
```

### `fly_axis_confidence.txt`

```
keypoint                 median      min      max
headTop                   0.920    0.584    0.963
thoraxCenter              0.881    0.634    1.006
abdomenTop                0.863    0.710    0.975
abdomenCenter             0.879    0.491    0.964
genitalia                 0.893    0.504    0.979
abdomenLeft               0.890    0.651    1.049
abdomenRight              0.835    0.547    1.019
wingLeft                  0.966    0.441    1.011
wingRight                 0.772    0.021    0.931
abdomenLowerLeft          0.910    0.619    1.016
abdomenLowerRight         0.882    0.455    0.991
forlegLeftJoint1          0.856    0.292    0.994
forlegLeftJoint2          0.788    0.110    0.985
forlegLeft                0.762    0.242    1.042
forlegRightJoint1         0.815    0.450    0.952
forlegRightJoint2         0.778    0.170    0.979
forlegRight               0.590    0.144    1.005
midlegLeftJoint1          0.881    0.542    1.022
midlegLeftJoint2          0.944    0.605    1.012
midlegLeft                0.911    0.230    1.041
midlegRightJoint1         0.867    0.345    1.022
midlegRightJoint2         0.865    0.019    0.999
midlegRight               0.921    0.101    1.006
hindlegLeftJoint1         0.794    0.083    1.065
hindlegLeftJoint2         0.867    0.393    1.021
hindlegLeft               0.686    0.231    0.990
hindlegRightJoint1        0.821    0.285    0.960
hindlegRightJoint2        0.877    0.153    1.015
hindlegRight              0.795    0.176    1.013
```

### `fly_rot_confidence.txt`

```
keypoint                 median      min      max
headTop                   0.874    0.382    1.051
thoraxCenter              0.911    0.628    0.983
abdomenTop                0.893    0.737    0.950
abdomenCenter             0.862    0.553    1.026
genitalia                 0.896    0.436    1.002
abdomenLeft               0.804    0.545    0.936
abdomenRight              0.866    0.399    1.002
wingLeft                  0.934    0.220    1.078
wingRight                 0.683    0.165    0.984
abdomenLowerLeft          0.853    0.645    1.003
abdomenLowerRight         0.857    0.460    0.969
forlegLeftJoint1          0.852    0.114    0.942
forlegLeftJoint2          0.808    0.148    1.000
forlegLeft                0.757    0.229    0.973
forlegRightJoint1         0.865    0.347    0.977
forlegRightJoint2         0.760    0.291    0.981
forlegRight               0.646    0.206    1.010
midlegLeftJoint1          0.929    0.539    1.022
midlegLeftJoint2          0.951    0.650    1.001
midlegLeft                0.935    0.428    1.012
midlegRightJoint1         0.915    0.450    1.005
midlegRightJoint2         0.882    0.015    1.013
midlegRight               0.886    0.174    1.026
hindlegLeftJoint1         0.804    0.279    0.971
hindlegLeftJoint2         0.828    0.363    1.027
hindlegLeft               0.820    0.130    0.981
hindlegRightJoint1        0.870    0.372    0.932
hindlegRightJoint2        0.937    0.362    0.990
hindlegRight              0.854    0.051    1.064
```

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
