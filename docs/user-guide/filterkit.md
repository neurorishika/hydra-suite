# FilterKit

FilterKit is the dataset filtering tool, launched via `filterkit`.

## Purpose

Filter and curate datasets for quality and diversity before training.

## Launch

```bash
filterkit
```

## Workflow

1. Load a dataset directory, or load a video to curate whole frames.
2. Apply filtering criteria (quality, diversity, metadata).
3. Preview and validate the filtered subset.
4. Export the filtered dataset for training or analysis.

## Video sources

Choose **Load Video** to select a supported video file (`.mp4`, `.mov`, `.avi`,
`.mkv`, or `.m4v`). FilterKit applies the same temporal, quality, duplicate, and
diversity filters to whole frames. Processing exports selected frames as PNG files
in a sibling `<video-name>_filterkit_output/images/` folder. The original video is
never modified; `filterkit_video_manifest.json` records the source video, selected
frame indices, and settings used for the selection.
