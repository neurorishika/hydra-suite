# Publishing to PyPI

How to release new versions of hydra-suite to PyPI.

## Prerequisites

`build` and `twine` are included in dev dependencies. Install them with:

```bash
make install-dev     # conda workflow
# or
pip install -e ".[dev]"  # pip workflow
```

You also need a [PyPI account](https://pypi.org/account/register/) and an [API token](https://pypi.org/manage/account/token/) (unless using trusted publishing via GitHub Actions).

## Release workflow

### 1. Update the version

Edit `pyproject.toml`:

```toml
version = "1.1.0"
```

This is the single source of truth — `__init__.py` reads it via `importlib.metadata` at runtime.

### 2. Build

```bash
rm -rf dist/ build/
python -m build
```

This produces both a `.whl` (wheel) and `.tar.gz` (sdist) in `dist/`.

### 3. Inspect the wheel

```bash
# Check size (should be ~1-2 MB)
ls -lh dist/*.whl

# Verify bundled assets are included
unzip -l dist/*.whl | grep -E "(resources/brand|resources/configs|py\.typed)"

# Verify old brand/ directory is NOT included
unzip -l dist/*.whl | grep "^.*brand/" | grep -v "hydra_suite/resources"
```

### 4. Test on Test PyPI (first release or major changes)

```bash
twine upload --repository testpypi dist/*

# Test in a clean environment
python -m venv /tmp/hydra-test
/tmp/hydra-test/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    hydra-suite
/tmp/hydra-test/bin/python -c "from hydra_suite.paths import get_brand_icon_bytes; print('OK')"
```

### 5. Upload to PyPI

```bash
twine upload dist/*
```

### 6. Tag the release

```bash
git add pyproject.toml
git commit -m "release: v1.1.0"
git tag v1.1.0
git push origin main --tags
```

## Automated publishing (recommended)

Instead of manual uploads, use GitHub Actions with trusted publishing. This eliminates API tokens.

### Setup (once)

1. On PyPI, go to your project → Settings → Publishing → Add trusted publisher
2. Enter your GitHub repo owner, name, workflow filename, and environment name

### Workflow

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

permissions:
  id-token: write

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: release
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

After this, creating a GitHub release auto-publishes to PyPI.

## What goes in the wheel

The wheel includes:

- All Python source under `src/hydra_suite/`
- `hydra_suite/resources/brand/*.svg`, `*.png` — app icons
- `hydra_suite/resources/configs/*.json` — default presets
- `hydra_suite/resources/configs/skeletons/*.json` — skeleton definitions
- `hydra_suite/py.typed` — type checker marker

The wheel does **not** include:

- `src/brand/` (original brand assets with `.ai` source files — excluded via `pyproject.toml`)
- `configs/` at repo root (bundled copies are in `resources/`)
- `models/`, `training/` (user data, lives in user home directory)
- `tests/`, `docs/`, `legacy/`, `tools/`

## Dependency architecture

```
pyproject.toml [project.dependencies]     ← single source of truth for base deps
    ├── numpy, scipy, pandas, PySide6, ultralytics, ...
    └── platformdirs (for user data dirs)

pyproject.toml [project.optional-dependencies]
    ├── cuda: onnxruntime-gpu, cupy-cuda12x
    ├── mps: onnxruntime
    └── dev: pytest, black, flake8, mypy, ...

requirements-*.txt (developer workflow only)
    ├── torch, torchvision (+ --extra-index-url for GPU)
    ├── GPU-specific: tensorrt, onnxruntime-gpu, cupy-*
    └── -e .  ← pulls pyproject.toml deps automatically
```

**torch** is deliberately excluded from `pyproject.toml` because:

- GPU variants require `--index-url` which PEP 621 cannot express
- Listing `torch` in deps would pull CPU torch from PyPI, potentially overwriting a user's GPU install
- This is standard practice (ultralytics, timm, transformers all do the same)

When adding new dependencies:

- **Base deps** (needed by all users) → add to `pyproject.toml` `[project.dependencies]`
- **GPU-specific deps** (CUDA only) → add to `pyproject.toml` `[project.optional-dependencies]` AND to `requirements-cuda.txt`
- **torch-related** → add to `requirements-*.txt` only (cannot go in pyproject.toml)

## Version checklist

Before releasing:

- [ ] Version bumped in `pyproject.toml`
- [ ] Changelog updated
- [ ] `make format && make lint` passes
- [ ] `python -m pytest tests/` passes
- [ ] Wheel builds cleanly: `python -m build`
- [ ] Wheel assets verified: `unzip -l dist/*.whl | grep resources`
- [ ] Test install in clean venv works
