# Detection limits

HYDRA tracks at most **1024 animals per frame**. This single limit governs
three things at once:

- the number of detections stored per frame in the detection cache,
- the number of per-animal analyses (pose, head-tail, identity) per frame, and
- the number of animals N itself (arenas x animals per arena).

## What happens at the limit

- Setting more than 1024 animals is rejected when tracking starts: the CLI
  (`trackerkit track`) prints a one-line error naming the limit, and the GUI
  shows a "Detection limit exceeded" message instead of starting the run.
- If a frame has more than 1024 candidate detections, the 1024 best are kept
  (most confident for neural detectors, largest by area for background
  subtraction), a warning names the frame, and the end of the run reports how
  many frames hit the limit ("Detection limit reached"). Frames named there
  lost detections: raise the confidence or size filters, or crop the arena,
  if that matters for your data.

## Changing the number of animals reuses inference

Detections are stored independently of N, so you can change the number of
animals at replay time without re-running the detector or the per-animal
models; only tracking is recomputed.

## Changing a filter reuses detections

Changing a detection filter (confidence, size, aspect ratio, ROI, or NMS IoU)
reuses the stored detections and recomputes only the per-animal stages
(pose, head-tail, identity) for the detections that now pass.

Detections are extracted down to a confidence of **0.01**, so the confidence
slider can be moved anywhere in 0.01 to 1.0 without re-running the detector.
A threshold below 0.01 has no extra effect.

## Runs from before this change

Earlier runs silently capped the number of animals at **128** (since
2026-09-03) without any warning. If you tracked more than 128 animals with
those versions, the stored results were truncated: re-run the tracking to get
all animals.
