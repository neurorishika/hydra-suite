# Spec 1 — Standalone ViTPose Port + Numerical Parity

**Date:** 2026-07-16
**Status:** Design — awaiting review
**Parent:** [`2026-07-16-vitpose-backend-roadmap.md`](./2026-07-16-vitpose-backend-roadmap.md)
**Scope:** Spec 1 of 4. Model + weights + parity only. No repo integration.

## Goal

A native-PyTorch ViTPose (classic B/L/H and ViTPose+ MoE) that provably matches
upstream numerically, proven **before** any refactor of HYDRA Suite touches it.

**Why standalone, and why first:** every genuine unknown in the ViTPose project
lives here. Whether `padding=2` matters as claimed, whether an on-device UDP decode
matches cv2, whether MoE `fc2` shapes load — all answerable in isolation against an
objective oracle (COCO AP) that does not care about this repo. Specs 2–4 are
engineering with known outcomes that touch working code. See the roadmap's
"Why this ordering".

## Non-goals

Explicitly out of scope, deferred to Specs 2–4:

- No `PoseInferenceBackend` implementation, no backend class.
- No changes to `PoseRuntimeConfig`, `PoseConfig`, or `create_pose_backend_from_config`.
- No registry, no runtime layer.
- **Export recipe IS in scope** (added 2026-07-16): `vitpose/export.py` converts a torch
  ViTPose to ONNX / TensorRT / CoreML, with a parity test per artifact. This is the
  model-specific half and stays inside the leaf -- it plays exactly the role
  ultralytics' `model.export()` plays for the YOLO backend.
- **Runtime EXECUTION stays out** (Spec 2): ONNX sessions, TRT engine lifecycle, CoreML
  sessions, warmup/AMP/batching. Also out: the auto-export *caching wrapper*
  (`auto_export_vitpose_model`), which belongs in `backends/vitpose.py` at Spec 3
  alongside `auto_export_yolo_model` (`yolo.py:38`) and `auto_export_sleap_model`
  (`sleap.py:1353`).
- No training, no fine-tuning.
- No GUI, no PoseKit changes.
- No SLEAP changes.

## Architecture

### Placement and boundaries

Self-contained leaf subpackage:

```
src/hydra_suite/core/identity/pose/vitpose/
├── __init__.py
├── config.py       # variant table
├── model.py        # ViT backbone, blocks, attention, MoE FFN
├── heads.py        # classic deconv head, simple decoder
├── weights.py      # download, load, strict-load assertions
├── transforms.py   # bbox→center/scale, UDP affine warp, normalize
├── decode.py       # decode_udp_cv2 (oracle), decode_udp_torch (production)
└── export.py       # torch -> ONNX / TensorRT / CoreML recipe
```

**Imports only `torch`, `timm`, `numpy`, `cv2`. Nothing from `hydra_suite`.**
(`export.py` additionally imports `onnx`/`tensorrt`/`coremltools` for whichever
target it is asked for — still nothing from `hydra_suite`.)
It lives in the repo (testable and reviewable in place) but is wired into nothing —
`create_pose_backend_from_config` does not learn it exists until Spec 3. Being a
leaf, it cannot violate the dependency direction in CLAUDE.md:159-165.

Module split follows the ~500-line rule (CLAUDE.md:123), modelled on `yolo.py`
(293 lines), not `sleap.py` (1780).

### Backbone (`model.py`)

Plain, non-hierarchical ViT. **Absolute learned pos-embed only — no relative
position bias, no window attention in any variant.** L/H are dense global attention
at every layer. (`relative_position_bias_table` appears in some configs as dead
BEiT-inherited boilerplate; ignore it.) Pre-norm blocks, `LayerNorm(eps=1e-6)`,
GELU, `mlp_ratio=4`, `qkv_bias=True`.

**Trap 1 — patch embed padding:**

```python
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=16, padding=2)
```

Upstream computes `padding = 4 + 2 * (ratio//2 - 1)`, which is `2` for `ratio=1`
(all released checkpoints). This is **not** standard ViT `padding=0`. Output is
coincidentally the same grid — `floor((256 + 4 - 16)/16) + 1 = 16`, giving 16×12 =
192 tokens — so a stock `timm` patch embed **loads without a shape error** and
silently samples a shifted pixel grid. Hardcode `padding=2`.

**Trap 2 — pos-embed cls slot:**

`pos_embed` is `nn.Parameter(1, num_patches + 1, D)` — 193 entries, retaining the
MAE cls slot even though no `cls_token` module exists in the ViTPose backbone.
Applied as:

```python
x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
```

Patch pos-embeds **plus the cls pos-embed broadcast to every token**. Dropping the
second term changes outputs. Reproduce exactly.

**Forward:** patch_embed → +pos → N× Block → `last_norm` (LayerNorm) →
`permute(0,2,1).reshape(B, D, Hp, Wp)`. Returns a feature map `(B, D, 16, 12)`, not
tokens.

**Attention** is textbook: fused `qkv` Linear → reshape `(B,N,3,heads,-1)` →
`q*scale @ k^T` → softmax → `@v` → `proj`. SDPA may be substituted, but **not
before AP is validated** — swap it in only after Gate B passes, then re-validate.

### Variant table (`config.py`)

| | embed_dim | depth | heads | head_dim | part_features | drop_path | layer_decay |
|---|---|---|---|---|---|---|---|
| S | 384 | 12 | **12** | **32** | 96 | 0.10 | 0.80 |
| B | 768 | 12 | 12 | 64 | 192 | 0.30 | 0.75 |
| L | 1024 | 24 | 16 | 64 | 256 | 0.50 | 0.80 |
| H | 1280 | 32 | 16 | 80 | 320 | 0.55 | 0.85 |

**ViTPose-S uses 12 heads at dim 384** (head_dim 32), not the usual 6. Easy to get
wrong from habit. `drop_path`/`layer_decay` are recorded here for Spec 4; they are
inert in Spec 1 (eval only).

### Heads (`heads.py`)

Both are upstream's `TopdownHeatmapSimpleHead`; config selects. Input `(B, D, 16,
12)` → output `(B, K, 64, 48)`.

**Classic** (`num_deconv_layers=2, filters=(256,256), kernels=(4,4),
final_conv_kernel=1`):

```
ConvTranspose2d(D,   256, k=4, s=2, p=1, output_padding=0, bias=False) → BatchNorm2d(256) → ReLU
ConvTranspose2d(256, 256, k=4, s=2, p=1, output_padding=0, bias=False) → BatchNorm2d(256) → ReLU
Conv2d(256, K, k=1, s=1, p=0)
```

**Simple** (`num_deconv_layers=0, upsample=4, final_conv_kernel=3`):

```
ReLU(x) → F.interpolate(scale_factor=4, mode='bilinear', align_corners=False) → Conv2d(D, K, k=3, s=1, p=1)
```

The `ReLU` is applied **before** upsampling, inside upstream's `_transform_inputs`.
Easy to miss. `align_corners=False` matters — flipping it shifts keypoints by a
fraction of a heatmap cell, which the ×4 upsample and bbox-scale multiply amplify
into several image pixels.

### MoE / ViTPose+ (`model.py`)

**Only the FFN changes.** Attention, patch embed, pos embed, and norms are
byte-identical to classic.

```python
class MoEMlp:
    fc1     = Linear(D, 4D)
    fc2     = Linear(4D, D - part_features)      # shared branch
    experts = ModuleList([Linear(4D, part_features) for _ in range(6)])

def forward(x, indices):
    x = act(fc1(x))
    shared_x = fc2(x)
    expert_x = sum(experts[i](x) * (indices.view(-1,1,1) == i) for i in range(num_expert))
    return cat([shared_x, expert_x], dim=-1)     # → D
```

- **Routing is NOT learned.** `indices` is the *dataset index*, passed in from
  outside (`dataset_source`) and threaded through every block.
- The masked-sum-over-all-experts is a DDP workaround (all experts run, then get
  zeroed). For single-dataset inference we index the expert directly — a pure win,
  numerically identical, and it avoids 6× the expert-branch compute.
- **`num_expert=6`**, fixed across S/B/L/H. Order:
  `0=COCO, 1=AiC, 2=MPII, 3=AP-10K, 4=APT-36K, 5=COCO-WholeBody`.
- For B: `fc2: 3072→576`, each expert `3072→192`, concat → 768.
- **Per-dataset heads:** upstream `TopDownMoE` holds `keypoint_head` (COCO, 17ch)
  plus `associate_keypoint_heads` — 5 more independent classic decoders with
  `out_channels` 14/16/17/17/133. Backbone shared, heads not.

Spec 1 loads all heads to satisfy `strict=True`, but evaluates only the COCO head
(index 0).

### Weights (`weights.py`)

Source: **`nielsr/vitpose-original-checkpoints`** on HuggingFace (re-hosts the
originals; it is what HF's own conversion script downloads). Official links are
OneDrive-only and 403 to `curl` behind a JS/bot gate.

Available: `vitpose-b.pth`, `vitpose-b-simple.pth`, `vitpose_base_coco_aic_mpii.pth`,
`vitpose+_{small,base,large,huge}.pth`, `vitpose_small.pth`, `vitpose_small_up4.pth`.
**Gap: no non-plus L/H** — deferred; Spec 1 gates on B.

```python
sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
```

**`weights_only=True` is required, not optional.** We are deliberately sourcing
from a third-party re-host, which is exactly the threat model `torch.load`'s
`weights_only=False` default exposes: unpickling arbitrary objects permits
arbitrary code execution. These are plain tensor state_dicts, so it costs nothing.
Record each checkpoint's SHA256 on first download and assert it thereafter.

Key structure — prefixes `backbone.`, `keypoint_head.`, plus
`associate_keypoint_heads.` for ViTPose+. Head keys:
`keypoint_head.deconv_layers.{0,1,3,4}.*` (0/3 = ConvTranspose, 1/4 = BN) and
`keypoint_head.final_layer.{weight,bias}`.

We keep upstream's names. HF's rename map exists but is irrelevant to us — writing
our own loader means we never need it.

### Transforms (`transforms.py`)

1. **bbox → center/scale**: COCO xywh → center; aspect-fix against 192:256;
   `padding_factor=1.25`; `scale = size / 200.0` (`pixel_std=200`).
2. **Affine → 256×192**. Configs write `image_size=[192, 256]` as **[w, h]** — a
   classic off-by-transpose:
   ```python
   trans = get_warp_matrix(theta=rot, size_input=center * 2.0,
                           size_dst=np.array([192, 256]) - 1.0,   # note the -1
                           size_target=scale * 200.0)
   img = cv2.warpAffine(img, trans, (192, 256), flags=cv2.INTER_LINEAR)
   ```
   `get_warp_matrix` is the **UDP** path. Upstream's legacy `get_affine_transform`
   with its 3-point `_get_3rd_point` construction is the **non-UDP** path — do not
   use it.
3. **Normalize**: RGB, `x/255`, `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`.

### Decode (`decode.py`) — two implementations, bound by a test

**Trap 3 — UDP is unconditional.** Every ViTPose COCO config sets `use_udp=True` in
both the pipeline and `test_cfg`. UDP (Unbiased Data Processing, Huang et al. CVPR
2020) is two coupled changes: the warp/coordinate transform is defined on **unit
length = pixel spacing** (`size - 1`) rather than pixel count, and decoding uses a
Taylor/Hessian refinement on the log-blurred heatmap. **Encoding, warping, and
decoding must agree.** Mixing costs ~1–2 AP silently.

**`decode_udp_cv2` — the oracle, not production.** Faithful port of upstream
`post_dark_udp`:

```python
preds, maxvals = _get_max_preds(heatmaps)          # integer argmax
preds = post_dark_udp(preds, heatmaps, kernel=11)  # GaussianBlur → clip(0.001,50) → log → Taylor
```

`cv2.GaussianBlur(hm, (11,11), 0)` — OpenCV's `sigma=0` means *derive from kernel*:
`0.3*((11-1)*0.5 - 1) + 0.8 = 2.0`, exactly the training `sigma=2`. Upstream's
docstring confirms ("k=11 for sigma=2").

> **HF deviation, deliberately not followed.** HF's processor hardcodes
> `gaussian_filter(sigma=0.8, radius=5)` — a much narrower blur that does not track
> `kernel_size`. It perturbs only sub-pixel refinement (argmax is unchanged), so the
> error is bounded under a pixel, but it is a genuine unflagged departure with no
> comment or issue explaining it. HF's own conversion-script `allclose` asserts are
> against HF's recorded outputs, so they do not catch it. We follow mmpose.
> HF also warps with `scipy.ndimage.affine_transform(order=1)` rather than
> `cv2.warpAffine`, and consequently only asserts `atol=1e-1` on pixels. We use cv2.

**`decode_udp_torch` — production, device-resident.** The fixed Gaussian as a
depthwise conv; the Hessian solve batched on GPU. This is what makes the
no-GPU→CPU-roundtrip requirement achievable in Spec 2: the faithful cv2 decode
would otherwise pull heatmaps to host, blur in OpenCV, and push coordinates back —
on every inference, in the hottest loop.

**`transform_preds` back to image space**, UDP branch:

```python
scale = scale * 200.0
scale_x = scale[0] / (output_size[0] - 1.0)   # the -1 is UDP-only
scale_y = scale[1] / (output_size[1] - 1.0)
```
with `output_size = [48, 64]` (heatmap w, h).

**Flip test** (`flip_test=True` in all configs): `img.flip(3)` → forward →
`flip_back(heatmap, flip_pairs)` → average `(hm + hm_flipped) * 0.5`. With UDP,
**`shift_heatmap=False`** — do *not* apply the `[:, :, :, 1:] = [:, :, :, :-1]`
column shift. That shift is the non-UDP correction; applying both double-corrects.

### Export (`export.py`) — the recipe, not the runtime

**Added to Spec 1 on 2026-07-16.** Reading the existing backends showed that
`auto_export_yolo_model` (`yolo.py:38`) and `auto_export_sleap_model` (`sleap.py:1353`) are
**not the export** — they are the caching wrapper (signature, location, sidecar, staleness),
delegating the actual conversion to ultralytics' `model.export()` and SLEAP's exporter.
Nobody supplies that conversion for ViTPose, so we do, and it belongs in the leaf beside the
model it encodes.

| layer | YOLO | SLEAP | ViTPose |
|---|---|---|---|
| recipe (torch -> artifact) | ultralytics `model.export()` | SLEAP exporter | **`export.py` — HERE** |
| caching wrapper | `auto_export_yolo_model` | `auto_export_sleap_model` | `backends/vitpose.py` — Spec 3 |
| lazy trigger | `api.py:75` | `api.py:117` | `api.py` — Spec 3 |

`export.py` takes a live `nn.Module` and a destination path. It knows nothing about
`PoseRuntimeConfig`, artifact caching, or where models live on disk — that is the wrapper's
job at Spec 3, and putting it here would break the leaf.

**The recipe. Every item is a ViTPose-specific fact verified during this port:**

- **`model.eval()` is mandatory.** The classic head's two `BatchNorm2d` layers are the only
  stateful modules; exporting in train mode emits training-mode `BatchNormalization` and
  silently produces garbage. Eval also makes `DropPath` a no-op, which otherwise traces to
  a random node.
- **Fixed 256x192 input.** `pos_embed` is a `(1, 193, D)` parameter with no interpolation
  path, and constant-folding bakes the `[:, 1:] + [:, :1]` slice-add into a constant. A
  frozen graph cannot serve another resolution — and 256x192 is the only shape the
  checkpoints target.
- **Dynamic batch only**, via `dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}`.
  `forward` does `x.permute(0,2,1).reshape(b, c, hp, wp)` with ints from `.shape`, which
  trace to literals; without `dynamic_axes` the graph pins the batch size too.
- **opset 14+.** mmpose's exporter asserts `opset_version == 11`; that is an mmpose-era
  constraint, not a model one.
- **`SimpleHead` already uses explicit `size=(64, 48)`**, not `scale_factor=4`. Task 4 did
  this deliberately: `scale_factor` traces to a `Resize` with computed sizes and is the
  classic ONNX shape-mismatch source. Do not "simplify" it back.
- **MoE exports a single expert.** Upstream's masked-sum runs all 6 and zeroes 5 (a DDP
  workaround); Task 9's int `dataset_index` path already indexes directly. Export with a
  concrete index: numerically identical for single-dataset inference, 6x less expert-branch
  compute in the graph.
- **TensorRT builds FP32.** Decided 2026-07-16 following the SLEAP keypoint precedent
  (`sleap.py:420-421` "fp16 is deferred to preserve keypoint precision"; same rule at
  `compute_runtime.py:141-142`). OBB's `half=True` is the wrong analog for a keypoint model
  whose entire value is sub-pixel accuracy.
- **Pre/post-processing is NOT exported and must not be.** The affine warp (cv2), the UDP
  decode, and `transform_preds` stay outside the graph. Export the model only. A correct
  graph with a sloppy decode looks exactly like "ONNX broke my model" — which is precisely
  how this project's first Gate C failure presented.

## Verification

### Gate A — strict load

`strict=True`, no missing/unexpected keys, on:

1. `vitpose-b.pth` → classic backbone + classic head
2. `vitpose-b-simple.pth` → classic backbone + simple decoder
3. `vitpose+_base.pth` → MoE backbone + 1 + 5 heads

This is the architecture unit test — it catches Traps 1 and 2 instantly and for
free. Gate A(3) is a distinct assertion: a ViTPose+ checkpoint **will not** load
into a classic `ViT` module, since MoE `fc2` is `[D - part_features, 4D]` rather
than `[D, 4D]`.

### Gate B — decode parity

`decode_udp_torch ≈ decode_udp_cv2` on real heatmaps produced by an actual forward
pass (not synthetic Gaussians — those are too well-conditioned to exercise the
Hessian solve near flat or multi-modal peaks).

**Tolerance:** max absolute difference `< 1e-2` heatmap units on decoded keypoint
coordinates, asserted per-keypoint rather than averaged. Rationale: one heatmap
unit is 4 image px at 64×48 → 256×192, and bbox-scale multiplies further, so 1e-2
heatmap units is comfortably sub-pixel in image space while still being ~100×
tighter than the ~0.3 AP that a sigma-level decode error produces. Averaging would
hide a single badly-decoded joint, which is the failure mode we care about.

If the two decoders cannot be made to agree at this tolerance, that is a **finding,
not a nuisance** — it means the torch Gaussian or Hessian differs structurally from
cv2's, and the resolution is to fix the torch decode, never to loosen the bound.

The chain: cv2 anchors us to mmpose, this test anchors torch to cv2, Gate C
validates end to end.

### Gate C — COCO val AP

Within ~0.2 of published, full UDP + flip-test, measured **through the torch decode**:

| checkpoint | expected AP |
|---|---|
| `vitpose-b.pth` (classic) | 75.8 |
| `vitpose-b-simple.pth` (simple) | 75.5 |

**Diagnostic ladder** — if AP is off, the magnitude names the trap:

| deviation | cause |
|---|---|
| ~1 AP | UDP mismatch (warp/decode disagree) |
| ~0.3 AP | decode blur sigma |
| wildly off | patch padding or pos-embed |

**The person-detections file — RESOLVED 2026-07-16, Gate C is reachable.**
Published top-down AP is only reproducible against the *standard detection set*
(`COCO_val2017_detections_AP_H_56_person.json`), not ground-truth boxes, which give
different (higher) numbers.

Verified by actual download: there is **no** OpenMMLab mirror (`download.openmmlab.com/
mmpose/datasets/person_detection_results.tar` → 404) and **no** HuggingFace mirror
(dataset search, full-text search, and the ViTPose model repos all negative). The
canonical HRNet OneDrive link 403s exactly like the checkpoints. The **GoogleDrive
folder works via `gdown`**:

```
pip install gdown
gdown 1ygw57X-mh0QBfENB-U5DsuSauGIu-8RB      # the val2017 file directly
# or the whole folder:
gdown --folder https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk
```

**Pin these — no upstream checksum is published, so these are computed from a
verified-genuine download:**

| | |
|---|---|
| size | 16,383,781 bytes |
| SHA256 | `53ba0ad8d0fd461c5a000cd90797fa8c39cd8c38cd125125c0412626ff592d59` |
| MD5 | `d5289281a44400280199b9ebda263743` |
| contents | 104,125 detections, 3,893 unique `image_id`, all `category_id: 1`, scores 0.0–1.0 |

> ### ⚠️ Trap 4 — the dummy detections file
>
> GitHub code search surfaces exactly one repo vendoring this path
> (`HuuTranDuc/LiteHrnet` → `data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json`).
> It returns **HTTP 200 with syntactically valid JSON**, so it looks like a clean
> mirror. It is **a dummy**: 250,475 bytes, 1,000 boxes, one per image, **every
> score exactly 0.99**. Using it silently produces garbage AP — the same
> fail-quietly mode as Traps 1–3, but in the eval harness rather than the model.
> **Assert the SHA256 above before evaluating.** Do not fetch from GitHub.

**Contingency (not needed unless the GDrive link dies):** GT-box AP against a
self-consistent baseline — compare our port against HF's `transformers` VitPose on
an identical pipeline with ground-truth boxes. Loses the absolute target, but
relative agreement still catches Traps 1–3. If used, compare **backbone and head
outputs (pre-decode)** for a clean signal, since HF's decode carries the
`sigma=0.8` deviation documented above.

**AP evaluation library — RESOLVED: use `pycocotools`.** Proven numerically rather
than assumed: xtcocotools' default sigmas are `np.allclose` to pycocotools' COCO
sigmas, and the full 10-element `stats` vector is identical on the same GT/DT
(`AP=0.68707` from both). xtcocotools is a fork adding only `sigmas=None` and
`use_area=True`, which matter solely for CrowdPose/AIC (no `area` field) and
wholebody (custom sigmas) — neither applies to standard COCO 17-keypoint AP.

This also dodges a real blocker: **`pip install xtcocotools` fails on Python 3.13**
(PyPI's latest is 1.14.3 from 2023-10, wheels cp37–cp311 only, no macOS arm64 at
all; the sdist then breaks on PEP 667 — `setup.py` reads `locals()` after `exec`,
which stopped working in 3.13). It is only installable from git master
(`pip install --no-build-isolation git+https://github.com/jin-s13/xtcocoapi`, which
carries an unreleased 2026-01-07 setup fix). We avoid needing it. Note this
constrains us to our own eval loop — mmpose's `CocoMetric` module-level
hard-imports xtcocotools — which we want anyway.

### Gate D — artifact parity (added 2026-07-16)

Every exported artifact must produce the same heatmaps as the torch model it came from. Same
discipline as Gate B: torch is the oracle, the artifact is under test, the bound is asserted
per-element rather than averaged.

| artifact | runner | where | bound |
|---|---|---|---|
| ONNX | `onnxruntime` (CPU EP) | mac + mehek | max abs diff < 1e-4 |
| TensorRT (FP32) | `tensorrt` python API | mehek only | max abs diff < 1e-3 |
| CoreML | `coremltools` | mac only | max abs diff < 1e-3 |

Bounds differ deliberately: ONNX-on-CPU is the same math at the same precision and should be
near-exact; TRT and CoreML rearrange kernels and get more slack, but FP32 keeps them close.
If an artifact misses its bound that is a finding about the recipe — **do not loosen the
bound**, and do not change `src/` to chase it (Gates A/B/C stand).

**Heatmap parity is necessary but not sufficient.** A small heatmap delta can still move a
decoded keypoint, and AP is what users feel. So the TensorRT artifact — the deployment
target — additionally gets a **Gate C run through the exported engine**, expected at
75.8 +/- 0.2 like the native path. ONNX and CoreML get heatmap parity only; a full AP run per
artifact is not worth the wall-clock.

**Risk — CoreML.** `coremltools 9.0` warns `Torch version 2.11.0 has not been tested with
coremltools. Torch 2.7.0 is the most recent version that has been tested.` CoreML export may
simply not work on this torch. It is sequenced last for that reason, and BLOCKED there does
not invalidate ONNX or TensorRT.

### Dependencies to acquire

| what | how | notes |
|---|---|---|
| COCO val2017 images + `person_keypoints_val2017.json` | `images.cocodataset.org` | ~1GB, needs downloading |
| `COCO_val2017_detections_AP_H_56_person.json` | `gdown 1ygw57X-mh0QBfENB-U5DsuSauGIu-8RB` | 16,383,781 B; **assert SHA256 `53ba0ad8…`** (Trap 4) |
| checkpoints | `nielsr/vitpose-original-checkpoints` (HF) | `weights_only=True`; record SHA256 |
| `pycocotools` | PyPI | sufficient; **not** xtcocotools |
| `gdown` | PyPI | eval/dev only — must **not** become a runtime dep of the package |

These are all eval-time assets. None belongs in git (the detections JSON alone is
16MB). Store under `hydra_suite.paths` conventions or a gitignored fixtures dir; do
not use `Path(__file__).parents[N]` (CLAUDE.md:199-201).

## Open questions

- Whether HF's decode `sigma=0.8` is deliberate — no comment or issue found. Does
  not block us (we follow mmpose), but worth an upstream issue if we ever rely on
  HF outputs for comparison.
- Non-plus L/H checkpoints remain OneDrive-only. Out of scope for Spec 1, which
  gates on B — but it means L/H may need a browser-assisted fetch later, or a
  conversion from whatever HF has.
- ViTPose+ effective batch size: config says `samples_per_gpu=64`, GPU count is not
  in the config, so the commonly-cited 512 is inferred. Only matters in Spec 4.

## Resolved during design (2026-07-16)

- ~~Person-detections file reachability~~ → `gdown`, SHA256 pinned. Gate C stands as
  the definition of done; the GT-box contingency is unused.
- ~~`pycocotools` vs `xtcocotools`~~ → pycocotools, proven numerically identical on
  standard COCO keypoint AP. Also avoids xtcocotools' Python 3.13 install failure.
