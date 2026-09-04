# DetectKit: SAHI Calibration Profiles

> Experimental calibration. This is a measurement tool, not a training tool.
> It never changes a model's weights, and it never writes the tracking
> project's `REFERENCE_BODY_SIZE`.

SAHI (Slicing Aided Hyper Inference) tiles a frame before running your
detector, so small animals get more pixels per inference pass. Tiling has
knobs -- tile size, overlap, confidence, merge policy -- and the right values
depend on your model, your animals, and your images. Calibration measures
those knobs against your own labelled frames instead of asking you to guess.

## What calibration measures, and what it never touches

Calibration runs your existing, already-trained detection model across a
grid of tiling and confidence settings, scores each combination against
frames you have already labelled, and reports the trade-offs. That is all
it does:

- It does **not** train or fine-tune anything. The model weights it loads
  are read-only for the entire run.
- It does **not** write to `REFERENCE_BODY_SIZE` or any other tracking
  project setting. Its output is a set of named **profiles** attached to
  the model file itself (see [Saving, naming, and the primary
  profile](#saving-naming-and-the-primary-profile)); a tracking project
  only picks one up if you explicitly apply it in TrackerKit.
- It does **not** modify your labels, your dataset, or your images.

Everything it produces is a measurement report you can inspect, save,
discard, or re-run.

## The exhaustive-labels requirement

Calibration counts a prediction as a false positive if it does not land on
a label. If your labelled frames are missing a real animal -- one you
simply didn't get around to boxing -- every correct detection of that
animal is counted as an extra, dragging down measured precision for every
candidate setting.

This biases the whole sweep toward settings that are **too strict**:
calibration will reward higher confidence thresholds and tighter tiling
because those happen to suppress the "false positive" that was actually a
missed label, not because those settings are genuinely better. The wizard
requires you to affirm your evidence frames are exhaustively labelled
before it will run, because there is no way to detect this bias after the
fact -- the numbers look perfectly self-consistent either way.

If you are not confident a frame is exhaustively labelled, either finish
labelling it or leave it out of the evidence set.

## Why the held-out `val` split is the default

Calibration defaults to your training run's `val` split -- the frames the
model never took a gradient step on. If you calibrated against frames the
model was trained on, you would be measuring how well the model fits data
it has already memorized, which reports optimistically and does not
predict how it behaves on new footage.

If no `val` split has labelled frames, calibration falls back to `train`
and says so plainly in the evidence summary -- it will not silently
substitute one split for the other. If neither split has usable frames, it
falls back to a stratified sample drawn directly from your linked sources.

## The candidate grid, and its cost

Calibration does not sweep every conceivable tile size. It builds a fixed
candidate grid from your model's own training geometry (its trained image
size and aspect ratio), so every candidate is a tiling scheme that is at
least plausible for a model trained the way yours was.

Each candidate is evaluated at every point in a **confidence x merge**
grid: confidence from 0.05 to 0.95 in steps of 0.05 (19 values), against a
fixed merge policy swept across three IoU thresholds (0.3, 0.5, 0.7). That
means the true cost of a calibration run is the number of tiling
candidates times 19 times 3 -- not one inference pass per candidate.

The wizard shows the exact candidate grid and its estimated tile cost
before anything runs, along with an estimated duration for each candidate.
**These are estimates from tile counts, not measured timings** -- the
run itself is what actually measures wall-clock time (see [Timings are
local measurements](#timings-are-local-measurements)).

## The detection cap, and why setting it too low silently caps recall

"Max targets per frame" bounds how many detections are kept per frame
before merging across tiles. It exists so a pathological configuration
(tiny tiles, low confidence) cannot produce an unbounded number of
candidate boxes.

If you set this cap below the number of animals actually present in a
frame, calibration cannot report full recall for that frame **no matter
how good the underlying detector and tiling settings are** -- the excess
detections are simply discarded before scoring. This failure is silent:
nothing in the sweep signals that the cap, not the model, is the limiting
factor. The wizard prefills the cap from the busiest frame in your
evidence set, but if your animals are more crowded in the field than in
your labelled sample, raise it.

**The cap is a measurement condition, not a profile setting.** A saved
profile records the cap its row was measured at (under `measurement`), but
it does not restore it. In TrackerKit the equivalent knob, `MAX_TARGETS`,
is the tracking slot count -- derived from your arena count and animals
per arena -- and belongs to your experiment, not to a detection profile.
So a profile reproduces the tiling, confidence and merge settings it was
measured with, and its scores are valid *for the cap named in the results
table*. If your TrackerKit `MAX_TARGETS` is far below that cap, expect
fewer detections per frame than the frontier reported.

## How to read the frontier columns

Calibration reports one row per measured operating point (one tiling
candidate x one confidence x one merge threshold). The columns that matter:

- **Matched** -- predictions that overlapped a label above the 0.5 IoU
  floor, one-to-one.
- **Missed** -- labelled animals with no matching prediction.
- **Extra** -- predictions with no matching label (includes both genuine
  false positives and any duplicate that failed to claim its label because
  a better-overlapping prediction claimed it first).
- **Duplicate** -- predictions that cleared the match threshold against a
  label another prediction had already claimed. These remain counted as
  extras for precision, but are broken out separately because a cluster of
  duplicates usually means tiles are double-detecting across a seam,
  which is a merge-settings problem, not a confidence problem.
- **Precision / Recall / F1** -- the standard ratios computed from matched,
  missed, and extra.
- **Mean IoU** -- the average overlap quality of the matched pairs. A high
  F1 with low mean IoU means the model is finding the right animals but
  boxing them loosely.
- **Seconds/frame** -- measured wall-clock time for that candidate's
  tiling geometry, on this machine, with this evidence.

A point with `failed_reason` set produced no usable output at all (for
example, the tile budget was exceeded) and is excluded from any
recommendation, though it is still shown so you can see it was tried.

## The recommendation rule

Calibration can suggest a balanced operating point, but it never applies
one automatically -- you always review and choose. The rule it uses, shown
in the app and quoted here verbatim:

> Balanced rule: drop failed and undersampled points, keep the Pareto
> frontier of misses, extras and time, then take the fastest point whose F1
> is within 0.01 of the best and whose localization quality is at least
> 0.5.

If no point clears the eligibility floors (at least 60 matched instances
and mean IoU at least 0.5), calibration refuses to recommend anything
rather than post a misleadingly perfect score from a handful of lucky
matches. In that case, label a few more frames or widen the sweep.

## Saving, naming, and the primary profile

After a run, you choose which measured points to keep as named **profiles**.
A profile is written into a sidecar file attached to the model checkpoint
-- it travels with the model, not with any one tracking project. You give
each saved profile a name (for example, "fast-and-crowded" or
"high-precision-sparse") so you can tell them apart later when applying one
in TrackerKit.

Exactly one profile per model can be marked **primary**. The primary
profile is the one TrackerKit falls back to automatically if a
previously-applied profile id can no longer be found in the model's
sidecar (for example, after re-calibrating and removing the old profile).
Removing the current primary profile requires you to explicitly name its
replacement, or explicitly clear the primary designation -- there is no
silent fallback to "whichever profile happens to be first."

## Applying a profile in TrackerKit

When you select a model with saved SAHI profiles in TrackerKit's detection
panel, a profile selector appears alongside the tiling controls. Choosing
"Training geometry" resets tiling to the values the model was trained
with, with no calibration applied. Choosing a named profile fills in its
measured tile size, overlap, object-tile fraction, confidence, and merge
settings.

If you then hand-edit any of those fields after selecting a profile, the
selector switches to **`Custom (based on <name>)`** -- it keeps naming the
profile you started from, but makes clear the settings on screen are no
longer exactly what was measured. This is informational only; TrackerKit
tracks with whatever settings are actually in the panel, custom or not.

If a saved profile id can't be resolved (its sidecar entry was removed,
or you're opening a project saved from a different model version),
TrackerKit falls back to the model's primary profile if one is set, and
says so in the status line; otherwise it falls back to training geometry.

## Timings are local measurements

Every seconds-per-frame figure calibration reports -- in the candidate-cost
estimate, the results table, and the recommendation text -- is a
measurement taken on the machine that ran the calibration, against the
specific evidence frames it used. It is not a portable performance
guarantee. A different machine, a different runtime tier (CPU vs. GPU vs.
GPU-fast), a different frame resolution, or a different animal density can
all change the real cost of a given tiling configuration. Re-run
calibration if you change machines or hardware in a way that matters to
your workflow, rather than trusting a number measured elsewhere.
