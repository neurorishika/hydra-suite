# SAHI Settings (shared across TrackerKit and DetectKit)

SAHI (Slicing Aided Hyper Inference) cuts a frame into overlapping tiles,
runs the model on each tile, and merges the results. Small animals then get
more pixels per inference pass than they would in one downscaled frame.

Every SAHI surface in the suite uses the **same settings widget**:

- the TrackerKit detection panel (direct YOLO mode);
- the DetectKit inference-settings dialog;
- the DetectKit YOLO training dialog;
- the SAM3 training panel;
- the SAM3 semantic-escalation dialog;
- the SAM2 geometry-escalation dialog.

Labels, ranges, tooltips and value resolution are identical in all six. Each
host shows only the rows that apply to it (see [Which hosts show which
rows](#which-hosts-show-which-rows)).

## Vocabulary

| Word | Means | Where you see it |
|---|---|---|
| **SAHI** | The feature as a whole. | Group titles, checkboxes, `--sahi-profile`. |
| **slice** | The saved configuration and the model's metadata contract. | `slice_*` config keys, `SLICE_*` engine keys, the `.slice_meta.json` sidecar. |
| **tile** | One piece of the grid at run time. | "Tile size", "Tile overlap", "Tiles per call". |
| **scale** | One object scale: the animal's size as a fraction of the tile. | "Object scale", "Object scales". |

The suite does not use "patch", "window" or "stride" for any of these.

## The controls

### Enable checkbox

Turns tiling on or off. Its text depends on the host: "Enable sliced
inference (SAHI)" in TrackerKit, "Enable sliced inference" in the DetectKit
inference dialog, and "Enable sliced training + preview" in YOLO training.
The SAM3 training and escalation hosts have no checkbox. In escalation, an
object scale of 0 means no tiling (see below).

While the checkbox is unticked, the group collapses to just the checkbox:
every other row, the Advanced section, the profile picker and the
tile-layout preview are hidden (in TrackerKit, so are the tile-batch note and
the profile status line). Nothing is reset. Tick it again and every value,
including whether Advanced was open, comes back as you left it. The DetectKit
inference dialog shrinks and grows with the group.

### Profile

The calibration profile picker: "Training geometry", each calibrated profile
saved with the model, or "Custom". The row appears only in TrackerKit, and
only when the selected model's sidecar contains calibration profiles, and
only while sliced inference is on: to switch to a profile, tick the Enable
checkbox first, then pick it. If
you edit a profile-owned value, the picker switches to `Custom (based on
<name>)`. Derived values never cause this switch. See [SAHI calibration
profiles](detectkit-sahi-calibration.md).

### Tile strategy

| Label | Stored value | Tile size |
|---|---|---|
| Use model input size | `auto_model` | The model input size, so there is no resampling. |
| Fit to animal size | `auto_object` | body size ÷ object scale. |
| Custom tile size | `custom` | The width and height you enter. |

Configs, projects and sidecars store the value in the middle column, never the
label. The tooltip on each item shows that value.

### Object scale

The object scale is the animal's size as a fraction of the tile:
**object scale = body size ÷ tile size**. With "Fit to animal size", the
tile is therefore body size ÷ object scale. A larger scale gives smaller
tiles: more tiles per frame and more pixels per animal.

- **Inference hosts** show one value, "Object scale". Next to it, a
  display-only hint gives the animal's size in model-input pixels at the
  host's real input size, for example `≈ 102.4 px at 1024 px input`. The
  scale is stored as a fraction only. Hosts no longer convert it to pixels
  with a fixed 640 px input.
- **Training hosts** show a set, "Object scales" (for example
  `0.05, 0.1, 0.15, 0.2`). The builder generates tiles at every scale, which
  makes the model robust to scale. The SAM3 training panel also has
  "Single-scale object fraction", a fraction of the 1008 px SAM3 input. It is
  used when no scale set is listed, or when the strategy is not "Fit to
  animal size".
- **Escalation hosts** (SAM3, SAM2) show one value. `0` displays as "full
  frame (no tiling)".

The object scale is editable only with "Fit to animal size" in the YOLO
hosts. Under the other strategies it is disabled, not hidden.

### Body size

The typical longest side of one animal in source-image pixels. Tile sizing
uses it under "Fit to animal size". The value is **derived**, so a small
badge beside it names its source (see [Source badges](#source-badges)).

- The SAM3 and SAM2 escalation dialogs offer **Override**, which lets you
  edit a measured or stamped value. Unchecking it restores the derived
  value.
- In the DetectKit inference-settings dialog the body size is an ordinary
  field that you edit directly (badge `user`).
- In TrackerKit the body size is display-only. It comes from the model's
  stamp, the applied profile, or the saved config, and TrackerKit never
  edits it. This value is not the tracking `REFERENCE_BODY_SIZE`, and no SAHI
  path ever writes `REFERENCE_BODY_SIZE`.
- The training hosts show a note instead of a field: "Last build: X px,
  measured automatically from labels". Before the first build, the note
  says the size will be measured when the sliced dataset is built.
- An unknown body size (0) shows "unknown (model input tiles)" in the YOLO
  hosts. In escalation it shows "unknown (tiling off)" and stays editable,
  so you can type one in.

### Tile size (derived)

The width and height spins can be edited only with "Custom tile size". Under
the other strategies they are disabled, and a label beside them shows the
tile that will actually be used. For example, `→ 480 × 480 px (48 px ÷ 0.1)`
for one scale, or `→ 240–960 px over 4 scales` for a training set. A width
or height of 0 means "model input". The escalation hosts call this row
"Resolved tile" and show only the label.

### Tile overlap and the whole-animal minimum

Overlap is the fraction of a tile that neighbouring tiles share. An animal is
guaranteed to lie whole inside at least one tile only when the overlap is at
least the animal's share of a tile. Next to the overlap, the widget therefore
shows the **whole-animal minimum**: the largest object scale + 0.05 (outside
"Fit to animal size": body size ÷ tile side + 0.05), capped at 0.9.

- If your overlap is at or above the minimum, the widget shows a muted
  `≥ whole-animal minimum (X)` and nothing more.
- If your overlap is below the minimum, the widget shows "Below whole-animal
  minimum (X)" and a **Raise to X** button. Your value is kept until you
  click the button.
- If a calibration profile, the model's stamp, or a saved session set the
  overlap, a lower value is reported as `below whole-animal minimum (X) — set
  by <source>`, with no button. A measured overlap is not nudged.

**The minimum is a hint and is never applied automatically.** No host
derives or rewrites a saved overlap. TrackerKit's saved 0.2, a YOLO training
project's saved overlap, SAM3 training's 0.25 and SAM3 escalation's 0.5 load
exactly as saved. New YOLO training projects start at 0.25 — the
whole-animal minimum of the default scale set (0.20 + 0.05) — so they don't
open below their own minimum; TrackerKit's serving default stays 0.2. A
deliberately higher overlap is never nudged down. SAM2 overlap is fixed at
0.5 and shown disabled.

The overlap maximum is 0.9 everywhere except SAM3 training, which accepts up
to 0.99 to match its own contract.

### Advanced (collapsed)

The rows under Advanced depend on the host:

| Row | Meaning |
|---|---|
| Merge threshold | Overlap above which duplicate predictions from neighbouring tiles are merged. |
| Merge IoU (SAM3 escalation) | Polygon IoU above which two masks from neighbouring tiles are the same animal. |
| Seam margin (px) (SAM3 escalation) | Masks this close to a tile edge are treated as cut by the seam when merging. |
| Minimum retained object area (training) | Minimum fraction of a labelled object's area that must lie in a tile for the label to stay a full target. |
| Below the floor (training) | What happens to a fragment under that minimum. YOLO: "Drop fragment". SAM3: "Mark is_crowd". The training backend fixes this, so it is shown disabled. |
| Empty-tile sampling fraction (YOLO training) | Probability of keeping a background-only tile. |
| Keep empty tiles (SAM3 training) | Keep tiles with no labelled object as negative evidence. |
| Mix full frames (training) | Add unsliced full-frame examples so the model keeps global context. |
| Balance multi-scale training loss / Balance strength (YOLO training) | Normalize the loss so that a scale producing more tiles cannot dominate. 0.5 gives square-root balancing and 1.0 gives exact balance. |
| Tiles per call / Memory (TrackerKit tile memory budget) | Execution knobs that set how many tiles go to the detector in one call. |

## Which hosts show which rows

While sliced inference is on, constrained fields are **disabled, never
hidden**: a row is hidden only when the host has no such setting. The one
exception is the Enable checkbox: while it is unticked, everything but the
checkbox is hidden (see [Enable checkbox](#enable-checkbox)).

In TrackerKit the settings use a **compact layout** so they fit the side
panel without sideways scrolling:

- Related fields share a row: Profile | Tile strategy, Object scale | Body
  size, Tile size | Overlap, and (under Advanced) Tiles per call | Memory. A
  field whose partner is not shown (for example, no profile row) stands alone
  at the left. The body size's source badge (`profile`, `stamped` ...) stays
  next to its field; the tile size's source follows the resolved tile size
  on the summary line, in brackets (`(derived)`, `(profile)` ...). Hover a
  Profile or Tile strategy box to see the selected item's full name.
- The derived values are one muted summary line under the fields, for
  example `→ 480 × 480 px (derived) · ≈102 px at 1024 · ≥ whole-animal minimum (0.15)`:
  the resolved tile size, the object's size at the model input (Fit to animal
  size only) and the overlap check. The line shortens when the panel is narrow;
  hover it for the full text. A below-minimum warning keeps its orange colour
  and its **Raise to X** button on that line and is never shortened.
- The tile-batch note sits under Tiles per call | Memory.
- The tile-layout preview sits **below** the controls at a fixed, shorter
  height with a two-line caption.

The DetectKit dialogs keep one field per row and the preview beside the
controls.

| Row | TrackerKit | DetectKit inference | YOLO training | SAM3 training | SAM3 escalation | SAM2 escalation |
|---|---|---|---|---|---|---|
| Group title | *(inside the YOLO group)* | Sliced inference (SAHI) | Sliced dataset / inference (SAHI) | Tiling (SAHI geometry) | Tiling (SAHI) | Tiling (SAHI) |
| Enable checkbox | yes | yes | yes | – | – | – |
| Profile | when the model has profiles | – | – | – | – | – |
| Tile strategy | yes | yes | yes | yes | – (always fit to animal) | – (always fit to animal) |
| Object scale(s) | one | one | set | set + single-scale fraction | one (0 = full frame) | one (0 = full frame) |
| Body size | display only | editable (`user`) | note (measured at build) | note (measured at build) | Override | Override |
| Tile size / Resolved tile | yes | yes | yes | yes | label | label |
| Tile overlap | yes | yes | yes | yes (max 0.99) | yes | fixed 0.5 |
| Tile-layout preview | below the controls (compact) | beside | beside | beside | – | – |
| Advanced | Tiles per call, Memory (tile memory budget) | Merge threshold | Merge threshold, min area, below the floor, empty-tile fraction, mix full frames, loss balance | min area, below the floor, keep empty tiles, mix full frames | Merge IoU, Seam margin | – |

TrackerKit's merge policy, metric and threshold belong to the model's
profile, so the panel does not show them. The **Calibrate** buttons sit next
to the widget in TrackerKit and in the escalation dialogs, not inside it.

## Source badges

| Badge | The value came from |
|---|---|
| `user` | You entered it. |
| `override` | Your override of a derived value. |
| `profile` | The selected calibration profile. |
| `stamped` | The model's training stamp in its sidecar. |
| `project` | The project's sliced-training reference body size. |
| `dataset` | A measurement of the project's labelled objects. |
| `config` | The saved session or advanced configuration, not the selected model. |
| `derived` | A computation from the settings above, such as a tile size. |
| `default` | The backend default. |

## How values resolve

All hosts use one rule set, defined in `utils/tiling_spec.py`.

- **Object scale at inference:** calibration profile → the model's stamped
  scale → the backend default (0.15 for YOLO).
- **Body size:** override → the project's label measurement (DetectKit) →
  the model's stamp. In DetectKit, the label measurement wins over the
  stamp. TrackerKit shows the stamp or the applied profile's value (see
  below).
- **Tile size:** always computed from the strategy, body size and scale,
  except with "Custom tile size".
- **Overlap:** your saved value, with the whole-animal minimum shown as a
  hint (see above).

Per host:

- **TrackerKit:** the profile you pick, else the model's primary profile,
  else the model's training geometry. A session saved with explicit
  settings restores those settings (badge `config`). The same ladder runs
  from `trackerkit track`, and `--sahi-profile` picks a profile explicitly
  (see [SAHI calibration profiles](detectkit-sahi-calibration.md)).
- **DetectKit preview:** follows the model's sidecar the way TrackerKit
  would for a fresh config. Geometry, scale, overlap, tile size and merge
  settings come from the model's primary profile, else its training
  geometry. The project keeps three things: the enable toggle, the input
  size (`imgsz_obb_direct`), and the body size (the label measurement
  first, the stamped body second). If you set values in the inference
  settings dialog, they win outright. If the model has no sidecar, the
  preview uses the project's settings. The status bar names the source
  ("profile 'X'", "training geometry", "inference-settings override" or
  "project settings").
- **SAM3 escalation:** the settings you last accepted for that model
  variant → its saved calibration → the model's stamped scale → the
  starting guess of **0.05** at the project's body size. That body size is
  the sliced-training reference, else the median of your labels. The 0.05
  seed is the escalation starting point. It differs on purpose from the
  SAM3 *training* default of 0.055. Tiling is off only when no body size is
  known. `detectkit escalate sam3` resolves the same way, and
  `--tile-fraction` / `--reference-body-px` override it (see [SAM2 and SAM3:
  install and run](sam-install-and-run.md#running-from-the-command-line)).
- **SAM2 escalation:** accepted settings → the calibration → full frame.
  SAM2 is a stock model, so it has no stamp, and until it is calibrated it
  segments full frames.

### One operating scale at inference

A model trained on a scale set (for example `0.05, 0.1, 0.15, 0.2`) is
**served at one scale**, the operating scale. The operating scale is the
median of the set, unless a calibration profile picks another. Multi-scale
training is a robustness tool, and inference never fans out over every
scale, so serving cost stays at one pass per tile grid.

## Where SAHI settings are stored

- **The model's sidecar** sits next to the weights as
  `<model>.<ext>.slice_meta.json`, for example `best.pt.slice_meta.json`. It
  records the training geometry, which provides the stamped values, and the
  calibration profiles. YOLO and SAM3 publishes both write it. Keep it next
  to the model file whenever you copy or ship the model.
- **TrackerKit configs** store `slice_*` keys. `slice_geometry_mode` stores
  the value (`auto_object`), not the label.
- **DetectKit projects** store the training and preview settings in the
  project file.

Older sidecars and configs (pixel `target_sizes`, `tile_overlap`,
`tile_fraction` and similar) still load: they are translated on read.
