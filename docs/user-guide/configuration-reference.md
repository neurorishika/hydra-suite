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
| `reference_body_size` | Body-size anchor (pixels @ resize=1) | experiment-specific |
| `enable_size_filtering` | Enable area filtering | bool |
| `min_object_size_multiplier` | Lower size bound vs body area | `0.1 - 5.0` |
| `max_object_size_multiplier` | Upper size bound vs body area | `0.5 - 10.0` |
| `subtraction_threshold` | Foreground threshold for BG subtraction | `0 - 255` |
| `background_learning_rate` | Adaptive background speed | `0.0001 - 0.1` |
| `yolo_model_path` | YOLO model file path | valid model path |
| `yolo_confidence_threshold` | Detector confidence gate | `0.01 - 1.0` |
| `yolo_iou_threshold` | IOU overlap threshold | `0.01 - 1.0` |
| `yolo_device` | Compute device selector | `auto/cpu/cuda:0/mps` |

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

## Configuration Behavior Notes

- Not all keys affect both detection modes.
- Temporal thresholds and velocities depend on FPS.
- Some features are runtime-only and not relevant for headless use.
