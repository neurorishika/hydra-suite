# ClassKit

ClassKit is the classification and embedding toolkit, launched via `classkit`.

## Purpose

Build identity classifiers from animal crops using embedding models, clustering, and active learning.

## Launch

```bash
classkit
```

## Workflow

1. Create or open a project with source image directories.
2. Ingest and embed crops using a backbone model.
3. Cluster embeddings and visualize with UMAP.
4. Label identity classes manually or via AprilTag auto-labeling.
5. Train a classification head and evaluate results.
6. Export labeled datasets for downstream use.

## Key Features

- Embedding extraction with configurable backbone models
- UMAP-based dimensionality reduction and visualization
- FAISS-powered similarity search and clustering
- AprilTag auto-labeling for marker-based identity assignment
- Active learning for efficient labeling
- Export to Parquet/CSV, ImageFolder, and Ultralytics classification formats

## Approving Predictions While Labeling

With a trained model loaded, labeling mode shows **Approve** / **Reject** (default keys `+` / `-`) for any image that has a proposal: either a staged unverified machine label or the live model prediction. No review-mode staging is needed.

- **Approve** stores the proposed label as a human-verified label and moves on.
- **Reject** asks for the correct label, stores that, and moves on.
- Images that already have a verified label, and `unknown` predictions, have nothing to approve.

## Reviewing Labels

**Review Labels…** (under the label buttons in labeling mode) opens a grid of every image that carries a label, one label at a time, to find and remove mislabels.

- Pick a label in the list on the left (or step with ↑ / ↓). Each entry shows its image count and, with a model loaded, how many of its images the model disagrees with (⚠).
- With predictions loaded, images are sorted by the model's probability for the assigned label, **lowest first**, so likely mislabels are on the first pages. Each tile shows that probability, plus the model's own guess in orange when it disagrees. Without a model, images are in project order.
- Click tiles, or press their keys (`1`–`9` for 3 × 3, `1234 / QWER / ASDF / ZXCV` for 4 × 4), to select them. **Select page** (Ctrl+A) toggles the whole page. Then **Remove label** (Del, or Backspace / the Mac delete key). Selections apply to the current page only and are dropped when you change page, grid size or label, so nothing off screen is removed.
- Removing only clears the label: the image goes back to unlabeled, and the change is saved right away. **Undo** (Ctrl+Z / Cmd+Z) restores the last removal exactly, jumping to the page that shows it, including whether it was an unverified machine label.
- **All labels / Human-verified / Machine (unverified)** filters by provenance; machine labels carry an **M** badge. **Model disagrees only** hides images where the model agrees.
- Pages: PgUp / PgDn or ← / →.

## Training Augmentation Defaults

The training dialog's *Space and Augmentations* tab starts with conservative
defaults tuned for canonical animal crops:

| Setting | Default | Why |
|---|---|---|
| Horizontal / vertical flip | 0 | Left/right are class labels for heading models; use *label expansion* for label-aware mirroring. |
| Hue jitter | 0.01 | Colour-tag identity depends on hue; keep it minimal. |
| Saturation / brightness / contrast | 0.10 | Mild lighting variation. |
| Scale jitter | 0.20 | Window size varies 0.8x-1.2x about the crop centre, so the model tolerates the animal appearing larger or smaller. |
| Aspect ratio jitter | 0.20 | The crop window's width:height ratio varies 0.83x-1.2x at constant area. The crop centre never moves and the animal is never stretched. |

Scale and aspect jitter apply to train samples only. For YOLO-classify they are
written as extra pre-fitted copies per training image
(`canonical_aug_copies`, default 3); validation and test images stay clean.

## Active Learning Batches

*Build Batch* (Active Learning panel) fills a labeling batch from six slots.
Slots draw from a shrinking pool, so no image is picked twice and the batch
always reaches the requested size; a slot with no signal hands its share to
uncertainty.

| Slot | Default | Picks |
|---|---|---|
| Uncertainty | 30% | Highest entropy. |
| Diversity | 25% | k-center / farthest-first over the remaining unlabeled images. |
| **Errors** | 20% | Unlabeled images most similar to images the model currently gets **wrong**. |
| **Rare clusters** | 10% | Images sampled with a bias toward small, under-labeled clusters. |
| Representative (or *Balance labels*) | 10% | Dense, low-coverage clusters (or under-represented classes). |
| Audit | 5% | Random + high-disagreement clusters. |

**Errors.** A known error is a human-verified label that disagrees with a real
model prediction (abstentions such as `unknown` do not count, and unverified
machine labels are excluded so the model's own guesses are never treated as
ground truth). Unlabeled images are ranked by cosine similarity of their
embeddings to each error, with a small bonus when they receive the same wrong
prediction, then taken round-robin across the errors, most confidently wrong
first, so every failure is represented rather than one dominating. Candidates
are tagged `error_neighbors near error #N (true->pred)` in the candidate table.
Multi-head models compare composite labels, the same strings the UI shows.
Verified labels the model was **trained on** also count, so run it on
held-out labels (or after a fresh training run) for the most informative errors.

**Rare clusters.** Each unlabeled image gets weight
`cluster_size ** -alpha * (1 - label_coverage)` and the slot is drawn without
replacement, so a 20-image cluster is far likelier to be picked from than a
2,000-image one, and stops being favoured once it is labeled. *Rarity* is
`alpha`: 0% is proportional to size (no bias), 100% gives every cluster the same
expected share. Cluster noise points count as rare. If the *Errors* and *Rare
clusters* shares push the recipe above 100%, every slot is scaled down evenly.

### Combination balancing (multi-factor schemes)

The **Combos** share (default 15%) targets factor combinations that are rare or missing in the labeled set, e.g. `blue_pink` with no labels.

- It works on **per-factor predictions** (head slices for multi-head models, marginals over composite classes for flat models), so it can surface combinations the model has never output. An image predicted `blue` in factor 1 and `pink` in factor 2 is a likely `blue_pink` even if no `blue_pink` was ever labeled.
- Picks are greedy with diminishing returns: each pick lowers the weight of the combination it is expected to fill, so a batch spreads over several under-represented combinations instead of one.
- Combinations containing `unknown` are not targeted, and unverified machine labels do not count as labeled examples.
- Each candidate is tagged `likely <combo>` in the candidate table.

Default recipe: 25% uncertainty, 20% diversity, 20% errors, 15% combinations, 10% rare clusters, 5% representative, 5% audit.

### Grid Balance (CAPTCHA-style)

**Grid Balance…** in the Active Learning group is a fast way to balance labels once a model has produced predictions. Instead of choosing among every label for one image, you answer one yes/no question across a whole grid: *click every image that is `blue · pink`*.

- Combinations are visited **rarest first**. Each one is filled up to the label count of the most common combination, or up to **Cap** if one is set. When a combination reaches its target, the dialog moves on by itself.
- The grid mixes three kinds of candidates: the most likely members of the combination (about half the tiles), borderline images from the next band down, and a few random images to catch what the model misses and keep you attentive.
- **Clicking a tile labels it** with the shown combination (a verified human label, saved right away), and a new candidate takes its place. **Unclicked tiles get no label.** They are only skipped for this combination and are never shown for it again.
- **None match** (Enter) draws a fresh grid. **Next label** (N) skips to the next combination. **Undo** (Backspace / Ctrl+Z) reverts clicks one at a time, even across combinations, and puts the image back in its tile.
- Keys: `1`–`9` for the 3 × 3 grid; `1234 / QWER / ASDF / ZXCV` for the 4 × 4 grid.
- The **Examples** strip shows labeled images of the current combination for reference.
- Only unlabeled images are shown. Images awaiting machine-label review are left for the review flow. Combinations containing `unknown` are included only when **incl. unknown** is checked.
- Predictions are not refreshed during a session; retrain afterwards and run it again if needed.

