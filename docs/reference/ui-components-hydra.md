# UI Components Reference (TrackerKit)

This reference describes the HYDRA Suite UI by tab, with practical guidance for selecting values.

## How To Use This Page

- Read one tab at a time in the same order you configure the app.
- Start from defaults, then tune only the controls that match your failure mode.
- Use one controlled video segment as your calibration clip before full runs.

## Global Layout

### Video / ROI Panel

| Feature | Role | How to use it |
|---|---|---|
| Arena bar | Arena-centric replacement for the old ROI toolbar; empty by default, switches to editing mode once a shape exists. | See "Arena and ROI Setup" below. |
| `+`/`-` zone buttons (Circle / Polygon) | Draw an inclusion or exclusion zone directly, with no separate mode switch first. | Press "+ Circle"/"+ Polygon" to add area, "- Circle"/"- Polygon" to subtract from the current arena. |
| Crop Video to ROI | Creates a cropped video for faster tracking. | Use when ROI occupies a small fraction of frame area. |
| Timeline + playback controls | Frame-level inspection before running tracking. | Scrub and inspect crossings/occlusions before choosing frame range. |
| Tracking frame range controls | Limit processing to specific interval. | Start after setup transients, end before unusable tails. |
| Zoom and pan tools | Pixel-level inspection for ROI and detection checks. | Drawing works correctly at any zoom level, so zoom in for precise placement before or during shape entry. |

### Arena and ROI Setup

The arena bar replaces the old ROI toolbar's include/exclude dropdown with an
arena-centric workflow. It has two states:

**Empty state.** Before any shape is drawn, the bar shows only "By default,
the whole video is used." next to two buttons: **+ Add Single Arena** and
**+ Add Grid of Arenas**. A video with no ROI at all is a valid, common case
-- tracking runs across the full frame.

**Editing state.** Once at least one shape exists (or a new arena has been
started), the bar switches to show:

- **Arena navigation** -- "Currently labelling: Arena N", with **< Previous**
  and **Next >** to step between existing arenas, and **+ Add new arena** to
  start a fresh, empty arena (it holds no shapes until you draw one).
- **Zone tools** -- **+ Circle**, **- Circle**, **+ Polygon**, **- Polygon**.
  There is no longer a separate include/exclude mode to set before drawing:
  the `+` buttons draw an inclusion zone, the `-` buttons draw an exclusion
  zone (subtracted from the current arena), both for the arena you are
  currently labelling. Left-click places points; right-click removes the
  last point. **Finish Shape** commits the shape once it is valid; **Undo**
  removes the most recent point or shape; **Clear Arena** removes every
  shape belonging to the current arena while keeping its arena number
  reserved (numbers are never renumbered, since they appear in the exported
  `arena_id` column).
- **Overflow menu (`...`)** -- **Add Grid of Arenas**, **Clear All**, and
  **Crop Video to ROI**.

Drawing works correctly at any zoom level: clicks map back to image
coordinates through the current zoom transform, so you can zoom in for
precision without breaking shape placement.

**One arena is exactly a plain ROI.** If you only ever draw a single arena
(with any number of include/exclude zones inside it), the exported tracking
CSV carries no `arena_id` column at all -- it behaves exactly like the ROI
masking of previous versions. The `arena_id` column only appears once a
second arena exists.

**Overlap warning.** Arenas must not overlap, because an animal in a shared
region cannot be assigned to a single arena. When arenas overlap:

- If the arena you are *currently labelling* is one of the overlapping
  pair, the warning banner names the conflicting arena(s), and **Previous**,
  **Next**, and **Add new arena** are disabled until you resolve it -- you
  cannot navigate away from an overlapping arena.
- If any other pair of arenas overlaps (not involving the one you're
  currently on), navigation stays enabled and the banner still explains
  which arenas conflict, but you can keep working elsewhere.
- Regardless of which arena you're on, **tracking itself is blocked** if
  *any* pair of arenas overlaps anywhere in the shape list -- the overlap
  must be resolved everywhere before you can start a run, not just for the
  arena you happen to be viewing.

### Grid Builder ("+ Add Grid of Arenas")

For layouts with many identical arenas (multi-well plates, arrayed dishes),
the grid builder generates a rectangular grid of arenas in one step instead
of hand-drawing each one. The generated shapes are ordinary `roi_shapes`
entries with sequential arena ids, individually editable afterwards like any
other arena -- the grid builder is a bulk-entry convenience, not a separate
persistent mode.

Controls:

- **Shape** -- Circle or Rectangle. Circle exposes a **Radius** field;
  Rectangle exposes **Width** and **Height** fields.
- **Top-Left Position X/Y** -- the origin of the grid (the position of
  arena 1) in image pixels.
- **Rows** / **Columns** -- grid dimensions. Their maximum is capped so that
  every generated arena's centre stays inside the frame given the current
  origin, spacing, and rotation.
- **X spacing** / **Y spacing** -- centre-to-centre pitch between columns
  and rows. Each has a floor: the tightest spacing that still guarantees
  adjacent arenas cannot touch or overlap for the chosen shape and size.
  Spacing controls are hidden entirely when there is only one row or one
  column, since spacing is meaningless in that case.
- **Rotation (about arena 1)** -- rotates the whole grid layout, in half
  degree steps from -45 deg to +45 deg, via either the slider or the spin
  box.

A live preview (drawn with the same renderer used for the main canvas, so it
cannot visually diverge from the real overlay) shows the resulting arenas
over the current reference frame as you adjust the controls, along with a
summary line ("N arena(s), ids first..last"). Accepting the dialog appends
the generated shapes to the existing `roi_shapes` list; it does not replace
arenas you already drew.

### Action Panel

| Feature | Role | How to use it |
|---|---|---|
| Preview Mode | Short run for quick parameter validation. | Use on a short calibration clip first. |
| Start Full Tracking | Full pipeline execution. | Run only after preview metrics and visual checks look stable. |
| Progress / FPS / ETA | Runtime visibility and performance monitoring. | Watch for sudden FPS collapse after parameter changes. |

## Tab 1: Setup

### Purpose
Define files, timing basis, and core runtime behavior.

### Key Controls

| Control | Role | Value selection guidance | Common failure mode |
|---|---|---|---|
| Input video | Source media path. | Use the exact file used for analysis and keep path stable. | Wrong file variant causes non-reproducible runs. |
| Acquisition FPS | Temporal scaling basis for velocities and durations. | Use true acquisition FPS, not assumed playback FPS. | Wrong FPS distorts motion thresholds and lifecycle timing. |
| CSV output | Final tabular output path. | Use per-video output folders. | Overwriting prior runs without versioning. |
| Config load/save | Persist and reuse tuning. | Save per organism/setup profile. | Reusing configs across incompatible setups. |
| Processing resize factor | Speed-vs-detail tradeoff. | Lower for speed, raise for tiny animals/dense scenes. | Over-downscaling misses small animals and shape detail. |
| Save confidence columns | Adds detector/assignment confidence to output. | Keep enabled when QA or active learning is planned. | Losing confidence diagnostics needed for troubleshooting. |
| Use cached detections | Reuse existing detection cache. | Enable while iterating post-processing/tracking logic. | Stale cache after detection-setting changes. |
| Visualization-free mode | Faster processing by reducing UI rendering work. | Enable for large batch runs after validation. | Expecting real-time visual feedback while enabled. |

## Tab 2: Detection

### Purpose
Configure animal detection quality and robustness to lighting/background changes.

### Key Controls

| Control group | Includes | How to choose values | Common failure mode |
|---|---|---|---|
| Detection backend | Method, compute device | Use background subtraction in controlled arenas; YOLO OBB in complex backgrounds. | Wrong method yields either noisy masks or missed animals. |
| Image adjustments | Brightness, contrast, gamma | Use minimal adjustments needed to stabilize separation. | Over-adjustment amplifies noise or clips detail. |
| Background model | Priming frames, adaptive background, learning rate, subtraction threshold | Increase priming in variable lighting; keep learning rate conservative. | Adaptive background absorbing animals over time. |
| Lighting stabilization | Enable, smooth factor, median window | Enable only when illumination drift is real and gradual. | Over-smoothing suppresses real scene changes. |
| Morphology and contours | Kernel size, min area, max contour multiplier | Tune to reject speckle while preserving true animal silhouettes. | Kernel too large removes small/close animals. |
| Conservative split and dilation | Split kernel/iters, merge threshold, extra dilation | Use when merged blobs occur frequently in close interactions. | Over-splitting single animals into fragments. |
| YOLO settings | Model, path, confidence, IoU, classes | Raise confidence for precision; adjust IoU for neighbor separation. | Low confidence floods downstream with false positives. |
| GPU/batching/TensorRT | Batch mode, batch size, TensorRT options | Increase only after baseline correctness is validated. | Throughput tuning before correctness creates hidden errors. |

## Tab 3: Tracking

### Purpose
Define association, motion prediction, and track lifecycle logic.

### Key Controls

| Control group | Includes | How to choose values | Common failure mode |
|---|---|---|---|
| Core assignment | Max targets, assignment distance, recovery distance, backward tracking | Scale distance terms by realistic body-size movement. | Distances too large increase identity swaps. |
| Kalman tuning | Process noise, measurement noise, velocity damping, maturity settings | Raise process noise for erratic motion; raise measurement noise for jittery detections. | Filters lagging or oscillating due to bad noise balance. |
| Assignment weights | Position, orientation, area, aspect ratio weights | Start with position-dominant weighting, then add shape/orientation constraints. | Overweighting weak features destabilizes matches. |
| Motion logic | Motion velocity threshold, instant flip, orientation limits | Set threshold above jitter floor and below real locomotion. | Noise interpreted as movement. |
| Lifecycle | Lost frames threshold, respawn distance | Increase lost-frame tolerance only for real occlusion durations. | Fragmentation (too low) or ghost tracks (too high). |
| Stabilization gates | Min detections to start, min detect frames, min tracking frames | Use to prevent premature tracking on unstable startup data. | Starting tracking before signal stabilizes. |

## Tab 4: Processing

### Purpose
Clean trajectories, interpolate gaps, and configure final visualization outputs.

### Key Controls

| Control group | Includes | How to choose values | Common failure mode |
|---|---|---|---|
| Post-processing gates | Min trajectory length, max velocity break, max occlusion gap | Use organism-specific motion bounds and occlusion duration priors. | Over-aggressive cleanup removing valid behavior segments. |
| Velocity z-score filter | Threshold, window, min velocity | Enable when sporadic spikes remain after tracking. | Filtering out true bursts in high-speed species. |
| Interpolation | Method, max gap | Keep gap small; linear first, spline only when warranted. | Hallucinated paths over long missing intervals. |
| Merge/refinement | Agreement distance, overlap frames | Tighten only when merges across neighbors are common. | Merging unrelated tracks under dense conditions. |
| Video output | Render toggle, labels/orientation/trails, marker/text/arrow sizing | Enable for QA/reporting; disable for speed-focused production. | High-cost renders slowing full runs. |
| Histograms | Enable and history window | Use medium windows for responsive but stable monitoring. | Window too large hiding short-term quality collapse. |

## Tab 5: Dataset Generation

### Purpose
Export selective training frames and metadata for downstream model training.

### Key Controls

| Control | Role | How to choose values | Common failure mode |
|---|---|---|---|
| Dataset name/class name | Dataset identity metadata. | Use stable naming by experiment and version. | Name collisions across exports. |
| Output directory | Export destination. | Keep dataset exports isolated per run. | Mixing multiple runs into one folder. |
| Max frames to export | Dataset size cap. | Start small, inspect quality, then scale. | Large low-quality exports reduce annotation efficiency. |
| Frame quality threshold | Candidate filtering gate. | Raise for precision, lower for diversity. | Over-filtering rare but important edge cases. |
| Diversity window | Temporal diversity control. | Increase to avoid near-duplicate adjacent frames. | Redundant frame-heavy exports. |
| Context frames | Include neighboring frames. | Enable for temporal tasks; disable for static keypoint tasks. | Unnecessary storage growth without modeling benefit. |
| Sampling strategy | Deterministic vs probabilistic selection behavior. | Use deterministic for reproducible baselines. | Inconsistent datasets across reruns. |

## Tab 6: Individual Analysis

### Purpose
Configure identity-focused crop generation and identity-method settings.

### Key Controls

| Control group | Includes | How to choose values | Common failure mode |
|---|---|---|---|
| Output configuration | Dataset name, output directory, image format, save interval | Use PNG for lossless quality when storage permits. | JPEG artifacts degrading downstream learning. |
| Crop geometry | Padding fraction, min/max crop sizing, crop multipliers | Use just enough context to include full animal geometry. | Over-padding introduces background bias. |
| Background handling | Background color selection | Keep background consistent across exports. | Mixed background conventions across datasets. |
| Identity method settings | Method, model file/confidence, tag family/decimate | Match method to physical markers in footage. | Method-marker mismatch causing low-confidence identities. |

## Practical Tuning Order

1. Setup (video, FPS, resize, output paths)
2. Detection (method, thresholds, morphology)
3. Tracking (assignment + lifecycle)
4. Processing (cleanup + interpolation)
5. Dataset/Individual analysis exports
