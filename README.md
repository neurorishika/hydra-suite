# HYDRA Suite

![HYDRA Suite Banner](docs/assets/banner.png)

Holistic YOLO-based Detection, Recognition, and Analysis Suite.

Multi-animal tracking, pose labeling, classification, detection training, dataset filtering, and interactive proofreading in one toolkit.

[![Docs](https://img.shields.io/badge/docs-online-0A66C2?style=flat-square)](https://neurorishika.github.io/hydra-suite/)
[![License](https://img.shields.io/badge/license-MIT-15803D?style=flat-square)](https://github.com/neurorishika/hydra-suite/blob/main/LICENSE)
![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Acceleration backends](https://img.shields.io/badge/acceleration-CPU%20%7C%20MPS%20%7C%20CUDA-111827?style=flat-square)
![Beta](https://img.shields.io/badge/status-beta-F59E0B?style=flat-square)

[Install](https://neurorishika.github.io/hydra-suite/getting-started/installation/) |
[User Guide](https://neurorishika.github.io/hydra-suite/user-guide/overview/) |
[Developer Guide](https://neurorishika.github.io/hydra-suite/developer-guide/architecture/) |
[API Reference](https://neurorishika.github.io/hydra-suite/reference/api-index/)

## What It Includes

HYDRA Suite is organized as one launcher plus a set of focused applications:

| Command | Purpose |
| ------- | ------- |
| `hydra` | Launcher and entry point |
| `trackerkit` | Multi-animal tracking |
| `posekit` | Pose labeling and pose-project workflows |
| `classkit` | Classification and embedding tools |
| `detectkit` | Detection model training and dataset tooling |
| `filterkit` | Dataset filtering and curation |
| `refinekit` | Interactive proofreading and correction |

The codebase shares one runtime model across apps, with support for CPU, Apple Silicon MPS, and NVIDIA CUDA depending on the selected workflow and installed providers.

## Quick Install

### pip (CPU)

```bash
pip install hydra-suite
```

### pip (GPU)

```bash
# Apple Silicon / MPS
pip install torch torchvision
pip install "hydra-suite[mps]"

# NVIDIA CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "hydra-suite[cuda]"
```

### Developer Install

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

The `make install*` targets also build and install the pinned Kronauer lab
AprilTag fork (`Social-Evolution-and-Behavior/apriltag` at
`c43a9b6e6b7dcfe0e7647a78eff6655a1d743c2c`) so AprilTag workflows use the lab
wrapper instead of the stock `apriltag` module.

The full installation matrix, platform notes, and troubleshooting live in the docs site:

- Getting Started: <https://neurorishika.github.io/hydra-suite/getting-started/installation/>
- Environments and Makefile: <https://neurorishika.github.io/hydra-suite/getting-started/environments/>

## Launch

```bash
hydra
trackerkit
posekit
classkit
detectkit
filterkit
refinekit
```

## Common Workflows

### Documentation

```bash
make docs-install
make docs-serve
make docs-build
make docs-check
```

### Development

```bash
make install-dev
make pytest
make format
make lint
make audit
```

### Packaging and Reference Material

```bash
make build
make techref-build
```

The repository also ships a publication-style technical reference source for the tracking algorithm under `docs/technical-reference/`.

## Docs

The docs site is the source of truth for setup, workflows, runtime behavior, and reference material.

- Documentation home: <https://neurorishika.github.io/hydra-suite/>
- User Guide: <https://neurorishika.github.io/hydra-suite/user-guide/overview/>
- Developer Guide: <https://neurorishika.github.io/hydra-suite/developer-guide/architecture/>
- API and CLI Reference: <https://neurorishika.github.io/hydra-suite/reference/api-index/>

## Project Status

The package metadata currently targets:

- Python 3.11, 3.12, and 3.13
- MIT licensing
- beta-stage scientific and research workflows

Primary package and console entry point names are current with the codebase:

- package: `hydra-suite`
- Python package: `hydra_suite`
- launcher command: `hydra`

## Links

- Docs site: <https://neurorishika.github.io/hydra-suite/>
- Source: <https://github.com/neurorishika/hydra-suite>
- Issues: <https://github.com/neurorishika/hydra-suite/issues>
- License: [LICENSE](LICENSE)
