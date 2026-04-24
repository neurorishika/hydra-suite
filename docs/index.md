# HYDRA Suite Documentation

![HYDRA Suite Banner](assets/banner.png)

Holistic YOLO-based Detection, Recognition, and Analysis Suite.

Multi-animal tracking, pose labeling, classification, detection training, dataset filtering, and interactive proofreading in one toolkit.

[![Source](https://img.shields.io/badge/source-GitHub-111827?style=flat-square&logo=github)](https://github.com/neurorishika/hydra-suite)
[![License](https://img.shields.io/badge/license-MIT-15803D?style=flat-square)](https://github.com/neurorishika/hydra-suite/blob/main/LICENSE)
![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Acceleration backends](https://img.shields.io/badge/acceleration-CPU%20%7C%20MPS%20%7C%20CUDA-111827?style=flat-square)

[Install](getting-started/installation.md) |
[User Guide](user-guide/overview.md) |
[Developer Guide](developer-guide/architecture.md) |
[Reference](reference/api-index.md)

!!! info "Use This Site as the Source of Truth"
    This docs site is the canonical guide for setup, workflows, feature behavior, and reference material.

## Applications

| Command | Purpose |
| ------- | ------- |
| `hydra` | Launcher and tool selector |
| `trackerkit` | Multi-animal tracking |
| `posekit` | Pose labeling and pose-project workflows |
| `classkit` | Classification and embedding tools |
| `detectkit` | Detection model training and dataset tooling |
| `filterkit` | Dataset filtering and curation |
| `refinekit` | Interactive proofreading and correction |

## Quick Navigation

- :material-rocket-launch: **Getting Started**

    Installation, first launch, and platform setup.

    [Open Getting Started](getting-started/installation.md)

- :material-play-circle: **User Guide**

    End-to-end workflow for tracking, post-processing, datasets, and identity analysis.

    [Open User Guide](user-guide/overview.md)

- :material-source-branch: **Developer Guide**

    Architecture, module map, data flow, extension points, and performance notes.

    [Open Developer Guide](developer-guide/architecture.md)

- :material-book-open-page-variant: **Reference**

    API docs, CLI docs, UI component references, FAQ, and changelog.

    [Open Reference](reference/api-index.md)

- :material-file-document-outline: **Technical Reference**

    Publication-style algorithm writeup plus LaTeX manuscript source for the current tracker.

    [Open Technical Reference](reference/technical-reference.md)

## Quick Start

=== "pip"

    ```bash
    # CPU
    pip install hydra-suite

    # Apple Silicon / MPS
    pip install torch torchvision
    pip install "hydra-suite[mps]"

    # NVIDIA CUDA
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install "hydra-suite[cuda]"
    ```

=== "Developer"

    ```bash
    # CPU
    make setup
    conda activate hydra
    make install

    # Apple Silicon
    make setup-mps
    conda activate hydra-mps
    make install-mps

    # NVIDIA CUDA
    make setup-cuda
    conda activate hydra-cuda
    make install-cuda CUDA_MAJOR=13
    ```

=== "Launch"

    hydra

    trackerkit         # Multi-animal tracking
    posekit            # Pose labeling
    classkit           # Classification / embedding
    detectkit          # Detection model training
    filterkit          # Dataset filtering
    refinekit          # Interactive proofreading

## Local Docs Workflow

=== "Serve locally"

    ```bash
    make docs-install
    make docs-serve
    ```

=== "Strict build"

    ```bash
    make docs-build
    make docs-check
    ```

## Package Scope

This documentation maps to the current package layout:

- `hydra_suite.launcher`
- `hydra_suite.trackerkit`
- `hydra_suite.posekit`
- `hydra_suite.classkit`
- `hydra_suite.detectkit`
- `hydra_suite.filterkit`
- `hydra_suite.refinekit`
- `hydra_suite.core`
- `hydra_suite.data`
- `hydra_suite.training`
- `hydra_suite.runtime`
- `hydra_suite.integrations`
- `hydra_suite.widgets`
- `hydra_suite.utils`
- `hydra_suite.paths`
- `hydra_suite.resources`
