# CLI Reference

## HYDRA Launcher

Entry: `hydra_suite.launcher.app:main`

### Command

- `hydra`

### Description

Opens the HYDRA Suite launcher, which provides a central hub to select and launch individual tools.

---

## TrackerKit

Entry: `hydra_suite.trackerkit.app:main`

### Command

- `trackerkit`

### Flags

- `--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}`
- `--no-file-log`
- `--log-dir <path>`
- `--version`

---

## PoseKit

Entry: `hydra_suite.posekit.gui.main:main`

### Command

- `posekit`

### Arguments

- `images` (optional positional): root folder of images
- `--out <path>`
- `--project <path>`
- `--new`

---

## ClassKit

Entry: `hydra_suite.classkit.app:main`

### Command

- `classkit`

---

## DetectKit

Entry: `hydra_suite.detectkit.app:main`

### Command

- `detectkit`
- `detectkit train --config /path/to/training.json`

The bare command launches the GUI. The `train` subcommand runs dataset
validation, derivation, training, run registration, and optional publishing
without importing Qt or requiring a display. See the
[headless training runbook](../runbooks/detectkit-headless-training.md) for the
configuration format and server examples.

---

## FilterKit

Entry: `hydra_suite.filterkit.app:main`

### Command

- `filterkit`

---

## RefineKit

Entry: `hydra_suite.refinekit.app:main`

### Command

- `refinekit`

---

## Examples

```bash
trackerkit --log-level DEBUG
posekit /path/to/images --out /path/to/project
posekit /path/to/images --new
```
