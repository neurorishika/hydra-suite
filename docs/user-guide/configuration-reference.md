# Configuration Reference

This page documents frequently used runtime keys from the TrackerKit GUI and saved configuration state.

## Full UI Control Reference

For a complete, control-by-control interface reference (question labels, value guidance, defaults, and failure modes), see:

- `/reference/ui-components-hydra/`

## Setup

| Key | Meaning | Typical Range |
|---|---|---|
| `file_path` | Input video path | valid file path |
| `csv_path` | Output CSV path | valid file path |
| `video_output_enabled` | Render visualization video | `true` / `false` |
| `video_output_path` | Output video path | valid file path |
| `fps` | Acquisition FPS used for temporal scaling | `1.0 - 240.0` |
| `resize_factor` | Processing downscale factor (background subtraction only; clamped to `1.0` for YOLO OBB) | `0.1 - 1.0` |

## Detection

| Key | Meaning | Typical Range |
|---|---|---|
| `detection_method` | `background_subtraction` or `yolo_obb` | enum |
| `reference_body_size` | Body-size anchor (pixels @ resize=1) — see the caveat below | experiment-specific |
| `enable_size_filtering` | Enable area filtering | bool |
| `min_object_size_multiplier` | Lower size bound vs body area | `0.1 - 5.0` |
| `max_object_size_multiplier` | Upper size bound vs body area | `0.5 - 10.0` |
| `subtraction_threshold` | Foreground threshold for BG subtraction | `0 - 255` |
| `background_learning_rate` | Adaptive background speed | `0.0001 - 0.1` |
| `yolo_model_path` | YOLO model file path | valid model path |
| `yolo_confidence_threshold` | Detector confidence gate | `0.01 - 1.0` |
| `yolo_iou_threshold` | IOU overlap threshold | `0.01 - 1.0` |
| `yolo_device` | Compute device selector | `auto/cpu/cuda:0/mps` |

### `reference_body_size` is detector-dependent

It is not a property of your animal. It is the **geometric mean** of the
detection box axes, `sqrt(major x minor)`, measured from *whichever detector
produced the boxes*. Two models on the same footage can disagree sharply:

| Detector | major | minor | aspect | `reference_body_size` |
|---|---|---|---|---|
| YOLO-OBB detect | 115.4 | 51.0 | 2.30 | **76.5** |
| Segmentation | 91.0 | 25.7 | 3.46 | **47.9** |

Those are real measurements from the same 489 frames. The box *lengths*
differ by only 21% — which is all you see when you eyeball an overlay — but
the *widths* differ 2x, and a geometric mean weights width exactly as much as
length. A segmentation model traces the silhouette; a detection model's box
swallows legs and antennae.

**Never carry the value across a model change.** Every body-scaled gate moves
with it, and the object-size window moves as the *square*. Carrying 49.83
onto the OBB model above turns the window into 390–3900 px² while that
model's median detection is 5852 px² — **1.8% of real detections survive**,
versus 93.5% at the correct 76.81. Tracking still runs, and the equivalence
gate still reports EQUIVALENT, because nothing checks that the fixture is
detecting anything. Re-run Auto-Set after any detector change and re-check
the body-scaled parameters.

## Tracking

| Key | Meaning | Typical Range |
|---|---|---|
| `max_targets` | Concurrent tracked targets | `1 - 200` |
| `max_assignment_distance_multiplier` | Matching gate radius | `0.1 - 20.0` |
| `recovery_search_distance_multiplier` | Recovery search radius | `0.1 - 10.0` |
| `enable_backward_tracking` | Run reverse pass | bool |
| `kalman_process_noise` | State evolution uncertainty | `0.0 - 1.0` |
| `kalman_measurement_noise` | Detection uncertainty | `0.0 - 1.0` |
| `kalman_velocity_damping` | Velocity retention factor | `0.5 - 0.99` |
| `lost_frames_threshold` | Track expiry threshold | `1 - 100` |

## Processing and Analytics

| Key | Meaning | Typical Range |
|---|---|---|
| `enable_postprocessing` | Enable cleaning pipeline | bool |
| `min_trajectory_length` | Fragment removal threshold | `1 - 1000` |
| `max_velocity_break` | Jump split threshold | experiment-specific |
| `max_occlusion_gap` | Gap tolerance before split | `0 - 200` |
| `interpolation_method` | `None/Linear/Cubic/Spline` | enum |
| `interpolation_max_gap` | Max fillable gap | `1 - 100` |
| `enable_histograms` | Runtime stats collection | bool |

## Dataset Generation and Identity

| Key | Meaning | Typical Range |
|---|---|---|
| `enable_dataset_generation` | Active learning export path | bool |
| `dataset_conf_threshold` | Frame-quality trigger sensitivity | `0.0 - 1.0` |
| `enable_identity_analysis` | Enable identity crop/export tools | bool |
| `identity_method` | identity mode selector | enum |
| `individual_output_format` | crop export image format | `png/jpeg` |

## Batch Session State

These fields live on the TrackerKit GUI session (`TrackerConfig`), not the
per-video engine config — they never get written into `<stem>_config.json`
and have no effect on tracking output, only on how a batch is launched. See
[TrackerKit command line](trackerkit-cli.md) for the equivalent `--gpus`
/ `--jobs` / `--threads-per-job` CLI flags.

| Key | Meaning | Typical Range |
|---|---|---|
| `batch_parallel` | Fan a batch out one child process per GPU instead of running sequentially in-process | `true` / `false` (default `false`) |
| `batch_parallel_jobs` | Max concurrent videos; `0` = one per selected GPU (or 1 without GPUs) | `0 - 64` (default `0`, auto) |
| `batch_parallel_gpus` | GPU selector: ordinals, ranges, UUID prefixes, or `auto` for every GPU `nvidia-smi` reports | string (default `"auto"`) |

## Configuration Behavior Notes

- Not all keys affect both detection modes.
- Temporal thresholds and velocities depend on FPS.
- Some features are runtime-only and not relevant for headless use.
