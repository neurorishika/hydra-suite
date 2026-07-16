# PoseKit Frame Mode Toggle — Design

## Context

FilterKit's `preserve_full_frames` feature (see `2026-07-01-filterkit-preserve-full-frames-design.md`) lets users export datasets where every individual on a selected frame is kept, not just the most-diverse crop. That dataset preserves full per-crop metadata (`frame_id`, `obb_corners`, canonicalization matrices) needed to eventually reconstruct full-frame multi-animal poses for bottom-up model training.

This design is the second step: PoseKit itself currently has no notion of "these images came from the same frame." All of its sampling and labeling flows (manual labeling, random selection, smart-select clustering, bulk move buttons) operate purely on individual images. To build a bottom-up-model-ready labeling set, a user needs every flow to optionally operate at frame granularity — selecting/labeling a frame pulls in every individual on it, not just one.

**Explicit non-goal:** actually reconstructing a full multi-animal frame image/pose from its constituent crops. That is separate, future PoseKit work (a dedicated "frame mode" pose-training toggle). This design only changes *which images get selected into the labeling set* and *how they get grouped for that purpose*.

## Frame Grouping

PoseKit has no existing per-frame metadata encoding. This design reuses FilterKit/MAT's identity-crop filename convention: `did<detection_id>.<ext>`, where `frame_idx = detection_id // 10000` and `det_idx = detection_id % 10000` (see `parse_identity_image_filename` in `naming.py`).

- A new helper module groups an image list into frames by parsing each filename with `parse_identity_image_filename`.
- Filenames that don't match (non-HYDRA sources, e.g. raw video frame exports) are each treated as a singleton frame-of-1. Frame Mode silently degrades to today's per-image behavior for these — the toggle is never disabled, it just has no effect on ungroupable images.
- Grouping is scoped per source: PoseKit's existing `DataSource`/source-management already tracks which images came from which source video/folder, so `frame_idx` values are only compared within a single source, never across sources.

## Config Schema

Add one field to `posekit/config/schemas.py`'s existing dataclass:

```python
frame_mode: bool = False
```

with the corresponding `to_dict`/`from_dict` updates. This is unrelated to and does not collide with the existing `mode: str` field, which governs the pose canvas's frame/keypoint editing mode — a different concept entirely.

## GUI: The Toggle

A prominent `QCheckBox` labeled **"Frame Mode"** is added at the very top of `PoseSourceBrowserPanel`, above the Labeling Set list — the first thing a user sees in the left panel. Tooltip text:

> "Frame Mode: sampling and labeling operations act on entire frames (all detected individuals together), not single crops. Required if you're building a dataset for bottom-up multi-animal pose models."

- Unchecked (default) = today's Individual Mode; all existing behavior is unchanged.
- Checked = Frame Mode; the behavior changes below take effect.
- State is persisted via `self.config.frame_mode`.

## Shared Commit Path

A single method, `_add_indices_to_labeling_set(indices: list[int], source_label: str) -> bool`, is the one place that actually mutates the labeling set. In Frame Mode, it expands `indices` to include every companion instance per frame, then — unless the caller has already disclosed the expansion via its own preview (see Smart Select below) — shows one `QMessageBox.question` confirmation before committing:

> "This will add {frame_count} frame(s) comprising {total_count} total instance(s), including {companion_count} companion instance(s), to the labeling set. Continue?"

Returns `True`/`False` for OK/Cancel; on cancel, nothing is added. `source_label` is used only for logging/status-bar text (e.g. "Random Selection", "Embedding Explorer"). Reused (not reimplemented) by manual labeling, both bulk-move buttons, Random Selection, and the Embedding Explorer's "Add to Labeling Set" button — five call sites, one mutation path, one confirmation wording.

## Behavior Changes

All of the following are gated on `self.config.frame_mode` being `True`, except item 2's underlying bug fix, which applies in both modes.

### 1. Manual Labeling (`save_current()`)

When a user labels an image that is not yet in the labeling set, and that image's frame has companions:

- Before committing the save, route the companion expansion through the shared commit path (`_add_indices_to_labeling_set`), which shows the confirmation dialog.
- **Cancel:** discard the keypoint edits just made (do not write the label file, do not add anything to the labeling set) — the frame stays exactly as it was.
- **Confirm:** save the keypoints as normal, then add every companion instance of that frame to the labeling set.

If the frame has no companions (singleton, e.g. non-HYDRA source), behavior is identical to today — no dialog shown.

### 2. Unlabeled → Labeling (`_move_unlabeled_to_labeling`)

**Behavior fix, applies in both modes:** today this button moves *every* unlabeled frame in the current source into the labeling set. This changes to moving only the frames **currently selected** in the Source Frames list, in both Individual and Frame Mode.

In Frame Mode specifically, the move goes through the shared commit path (`_add_indices_to_labeling_set`), which expands each selected frame to include its companion instances and gates the move behind the confirmation dialog.

### 3. Unlabeled → All (`_move_unlabeled_to_all`, the revert-to-source action)

Today this button removes selected/unlabeled frames from the labeling set, reverting them to the source pool. In Frame Mode:

- A frame is only reverted if **none** of its instances are currently labeled.
- A frame with at least one labeled instance is skipped entirely — none of its instances (labeled or not) are reverted.
- After the operation, a `QMessageBox.information` reports how many frames were skipped and why. This is a post-action notice, not a pre-confirmation, since the Frame Mode guard can only prevent data loss, never cause it.

### 4. Random Selection (`_add_random_to_labeling`)

In Frame Mode, `random.sample` draws from the set of candidate **frame IDs** (frames with at least one not-yet-labeled instance) rather than individual image indices. The existing count spinbox keeps its literal meaning — "N items to add" — now interpreted directly as N frames (no back-solving); its label/tooltip updates to say "frames" when Frame Mode is on. The chosen frame IDs are expanded and committed through the shared commit path.

### 5. Smart Select — Cluster Coverage Counting

The existing per-individual embedding computation and clustering (`cluster_embeddings_cosine`, cosine similarity over DINOv2/CLIP/etc. embeddings) is **unchanged** — critically, frame-level diversity is *not* computed by averaging companions' embeddings together, since that would blur exactly the signal Smart Select is meant to preserve.

Instead, a new frame-selection layer sits on top of the existing per-individual clustering output:

1. Cluster all candidate (not-yet-labeled) individuals into `K` clusters using the existing pipeline (`self.k_spin`, unchanged meaning and default).
2. Group individuals by frame (per the Frame Grouping section above).
3. Greedily select frames:
   - Maintain a set of "covered" clusters (initially empty).
   - Each round, score every not-yet-selected candidate frame by the number of *distinct, not-yet-covered* clusters among its individuals.
   - Pick the highest-scoring frame. Tie-break by (a) total distinct clusters spanned (covered or not), then (b) smallest `frame_idx`, for determinism.
   - Add all of that frame's clusters to the covered set.
4. Stop when either:
   - The frame budget (`self.n_spin`, "Frames to add" — already frame-count-shaped, reused as-is, meaning N frames directly with no back-solving) is exhausted.
   - All `K` clusters are covered *and* continuing would not improve coverage — in this case, keep picking by residual best-score (ignoring the "not-yet-covered" restriction) until the frame budget is used, so the full requested budget is spent rather than silently under-filling.
5. Selecting a frame pulls in every not-yet-labeled instance on it, added directly to `self.selected_indices` (no separate confirmation dialog — see UI Changes below for why).

#### 5a. SmartSelectDialog UI Changes

The dialog's controls and preview were built for the old per-individual stratified algorithm and don't represent the new greedy cluster-coverage algorithm:

- When `self.config.frame_mode` is on, show a read-only note near the top of the dialog: "Frame Mode is ON — results grouped by source frame." There is no dialog-local Frame Mode toggle; the dialog always reflects the single global setting.
- `self.min_per_spin` (min-per-cluster quota) and `self.strategy_combo` (centroid/centroid_then_diverse) are disabled and grayed out, with tooltip "Not used in Frame Mode — frame selection uses greedy cluster-coverage instead of per-cluster quotas." Both are meaningless for the new algorithm, which has no quotas or strategy choice.
- `self._preview()` dispatches to the new frame-selection algorithm (Section 5, steps 1–4) instead of `pick_frames_stratified()` when Frame Mode is on; clustering via `cluster_embeddings_cosine()` is unchanged either way.
- The preview list (`self.preview`) switches from one line per image to one line per selected frame: `[frame 0042] covers clusters {3,7} — 3 instances (1 labeled)`. This is a deliberate design choice: because Smart Select already requires the user to review this preview before clicking "Add," the expanded companion list showing here **is** the warning — no separate confirmation modal is shown when `_on_add()` runs, unlike the other four Frame Mode entry points, which have no comparable preview.

#### 5b. EmbeddingExplorerDialog "Add to Labeling Set"

No changes to the UMAP scatter plot, hover preview, or point-selection interaction (out of scope — this is a targeted fix to one commit action, not a visualization redesign). The dialog still treats points as individual images with no frame-aware rendering.

The only change: `self.btn_add_sel`'s handler, which today calls `self.accept()` with `self.selected_indices` (local image indices) unchanged, now routes its selection through the shared commit path (`_add_indices_to_labeling_set`) when Frame Mode is on. Since this dialog's selection summary shows only a raw count with no frame/companion breakdown, the confirmation dialog is shown here (unlike Smart Select's own Add button in 5a).

## Testing Approach

- Frame-grouping helper: unit tests for filename parsing (matching and non-matching cases), per-source scoping.
- Config schema: round-trip `to_dict`/`from_dict` test including the new field, default `False`.
- Each of the six behavior changes (manual labeling, both bulk-move buttons, Random Selection, Smart Select, Embedding Explorer) gets its own test exercising Frame Mode on vs. off, using synthetic multi-instance-per-frame fixtures (mirroring FilterKit's test fixture style: distinguishable images per cluster/frame so real clustering/greedy behavior is exercised, not degenerate all-filtered-out fixtures).
- Smart Select cluster-coverage greedy loop: a dedicated test with a small fixed embedding set where the optimal coverage-maximizing frame order is known ahead of time, asserting the selection matches.
- `_add_indices_to_labeling_set`: unit tests covering expansion correctness (companions pulled in, no duplicates) and the cancel path (nothing added) independent of any specific caller.
- SmartSelectDialog: test that `min_per_spin`/`strategy_combo` are disabled in Frame Mode, and that the preview text renders one line per frame with correct cluster/instance/labeled-count content.
- EmbeddingExplorerDialog: test that its "Add to Labeling Set" action, in Frame Mode, expands the raw point selection to companions and routes through the shared commit path (confirmation shown, cancel leaves selection uncommitted).
