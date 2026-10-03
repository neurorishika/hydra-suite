.PHONY: configure-mps-libs env-create env-create-cuda env-create-mps env-update env-update-cuda env-update-mps env-remove env-remove-cuda env-remove-mps install install-cuda install-mps install-apriltag-fork install-sam3-clip setup-sam3-train setup-sleap doctor install-dev configure-cuda-ort setup setup-cuda setup-mps test pytest test-cov test-cov-html clean docs-install docs-serve docs-build docs-quality docs-check techref-build techref-clean pre-commit-install pre-commit-autopep8 pre-commit-run pre-commit-update format format-check lint lint-fix lint-strict lint-report dead-code dead-code-fix dep-graph dep-graph-text type-check audit benchmark build publish publish-test help

# Environment names for different platforms
ENV_NAME = hydra
ENV_NAME_GPU = hydra-cuda
ENV_NAME_MPS = hydra-mps
# CUDA major: empty = detect from the NVIDIA driver (>=580 -> 13, else 12).
# Override with `make install-cuda CUDA_MAJOR=12`.
CUDA_MAJOR ?=
PYTEST ?= pytest
PYTHON_BIN = $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python)
PRE_COMMIT = $(if $(CONDA_PREFIX),"$(CONDA_PREFIX)/bin/pre-commit",pre-commit)

# Every install target is a thin wrapper around install.py -- the one
# cross-platform installer (Windows users run it directly). Python
# dependencies live ONLY in pyproject.toml; tested versions in constraints/.
# BOOT_PY runs the installer itself and may be any Python >= 3.9.
BOOT_PY ?= python3
INSTALL = $(BOOT_PY) install.py
CUDA_FLAG = $(if $(CUDA_MAJOR),--cuda $(CUDA_MAJOR),)
# Inside an activated env, install into it; otherwise install.py manages the env.
CURRENT = $(if $(CONDA_PREFIX)$(VIRTUAL_ENV),--target current,)

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# Step 1: create the conda env (python + ffmpeg + build tools only)
env-create:
	$(INSTALL) --target conda --tier cpu --env $(ENV_NAME) --create-only

env-create-cuda:
	$(INSTALL) --target conda --tier cuda $(CUDA_FLAG) --env $(ENV_NAME_GPU) --create-only

env-create-mps:
	$(INSTALL) --target conda --tier mps --env $(ENV_NAME_MPS) --create-only

# Step 2: install packages into the ACTIVATED env
install:
	"$(PYTHON_BIN)" install.py $(CURRENT) --tier cpu

install-cuda:
	"$(PYTHON_BIN)" install.py $(CURRENT) --tier cuda $(CUDA_FLAG)

install-mps:
	"$(PYTHON_BIN)" install.py $(CURRENT) --tier mps

# Individual installer steps (kept for existing muscle memory)
install-apriltag-fork:
	"$(PYTHON_BIN)" install.py $(CURRENT) --only apriltag

install-sam3-clip:
	"$(PYTHON_BIN)" install.py $(CURRENT) --only clip

configure-mps-libs:
	"$(PYTHON_BIN)" -m hydra_suite.runtime.macos_libomp

# Retired: ONNX Runtime now loads its CUDA libraries from the pip nvidia-*
# wheels (onnxruntime.preload_dlls), so there is no LD_LIBRARY_PATH hook to
# write. Kept as a name; runs the CUDA self-check instead.
configure-cuda-ort:
	"$(PYTHON_BIN)" -m hydra_suite.runtime.doctor --tier cuda

doctor:
	"$(PYTHON_BIN)" -m hydra_suite.runtime.doctor

# Optional sidecar envs
# The sidecar copies hydra-suite's source from the env it is run against, so
# run these from the ACTIVATED main env.
setup-sam3-train:
	"$(PYTHON_BIN)" install.py $(CURRENT) --tier cuda $(CUDA_FLAG) --with-sam3-train --only sam3-train

setup-sleap:
	"$(PYTHON_BIN)" install.py $(CURRENT) $(CUDA_FLAG) --with-sleap --only sleap

# =============================================================================
# ENVIRONMENT MAINTENANCE
# =============================================================================

env-update:
	$(INSTALL) --target conda --tier cpu --env $(ENV_NAME) --update

env-update-cuda:
	$(INSTALL) --target conda --tier cuda $(CUDA_FLAG) --env $(ENV_NAME_GPU) --update

env-update-mps:
	$(INSTALL) --target conda --tier mps --env $(ENV_NAME_MPS) --update

# Remove environments
env-remove:
	@echo "Removing CPU environment..."
	conda env remove -n $(ENV_NAME)

env-remove-cuda:
	@echo "Removing NVIDIA GPU (CUDA) environment..."
	conda env remove -n $(ENV_NAME_GPU)

env-remove-mps:
	@echo "Removing Apple Silicon (MPS) environment..."
	conda env remove -n $(ENV_NAME_MPS)

# =============================================================================
# TESTING & VERIFICATION
# =============================================================================

test:
	@echo "Testing package installation..."
	python -c "from hydra_suite.trackerkit.app import main; print('✅ Import successful')"
	@echo "✅ All tests passed!"

pytest:
	@echo "🧪 Running pytest..."
	$(PYTEST)

test-cov:
	@echo "🧪 Running pytest with coverage..."
	@if ! $(PYTEST) --help 2>/dev/null | grep -q -- '--cov'; then \
		echo "ERROR: pytest-cov is not available in the active pytest environment."; \
		echo "Current pytest: $$(command -v $(PYTEST) || printf 'not found')"; \
		echo "Install dev tools in the active environment:"; \
		echo "  make install-dev"; \
		echo "Or point Make at a pytest entrypoint that already has pytest-cov:"; \
		echo "  PYTEST=./.conda/bin/pytest make test-cov"; \
		exit 1; \
	fi
	$(PYTEST) --cov=src/hydra_suite --cov-report=term

test-cov-html:
	@echo "🧪 Running pytest with HTML coverage report..."
	@if ! $(PYTEST) --help 2>/dev/null | grep -q -- '--cov'; then \
		echo "ERROR: pytest-cov is not available in the active pytest environment."; \
		echo "Current pytest: $$(command -v $(PYTEST) || printf 'not found')"; \
		echo "Install dev tools in the active environment:"; \
		echo "  make install-dev"; \
		echo "Or point Make at a pytest entrypoint that already has pytest-cov:"; \
		echo "  PYTEST=./.conda/bin/pytest make test-cov-html"; \
		exit 1; \
	fi
	$(PYTEST) --cov=src/hydra_suite --cov-report=html --cov-report=term
	@echo "📊 Coverage report generated in htmlcov/index.html"

# =============================================================================
# MODEL BENCHMARKING
# =============================================================================

# Helper: run a command inside the CUDA conda env without activating it
CONDA_RUN_GPU = conda run -p $(shell conda info --base)/envs/$(ENV_NAME_GPU) --no-capture-output

benchmark:
	@echo "⏱  Running tier-based pipeline benchmark (CUDA env). Set BENCHMARK_CONFIG=path/to/config.json"
	$(CONDA_RUN_GPU) python tools/benchmark_pipeline.py $(BENCHMARK_CONFIG) $(BENCHMARK_ARGS)

clean:
	@echo "🧹 Cleaning Python cache files..."
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@echo "✅ Cleanup complete!"

# =============================================================================
# COMPLETE SETUP
# =============================================================================

setup:
	$(INSTALL) --target conda --tier cpu --env $(ENV_NAME) --create-only
	@echo "Next: conda activate $(ENV_NAME) && make install   (or just: python install.py)"

setup-cuda:
	$(INSTALL) --target conda --tier cuda $(CUDA_FLAG) --env $(ENV_NAME_GPU) --create-only
	@echo "Next: conda activate $(ENV_NAME_GPU) && make install-cuda   (CUDA auto-detected; CUDA_MAJOR=12|13 overrides)"

setup-mps:
	$(INSTALL) --target conda --tier mps --env $(ENV_NAME_MPS) --create-only
	@echo "Next: conda activate $(ENV_NAME_MPS) && make install-mps"

# =============================================================================
# PACKAGING & PUBLISHING
# =============================================================================

build:
	@echo "📦 Building wheel and sdist..."
	rm -rf dist/ build/
	python -m build
	@echo ""
	@echo "✅ Built:"
	@ls -lh dist/
	@echo ""
	@echo "Verify assets:"
	@unzip -l dist/*.whl | grep -c "resources/" | xargs -I {} echo "  {} resource files bundled"

publish-test: build
	@echo "📤 Uploading to Test PyPI..."
	twine upload --repository testpypi dist/*
	@echo ""
	@echo "✅ Uploaded to Test PyPI."
	@echo "   Test with: pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hydra-suite"

publish: build
	@echo "📤 Uploading to PyPI..."
	@echo "   ⚠️  This publishes to the REAL PyPI. Press Ctrl+C to cancel."
	@sleep 3
	twine upload dist/*
	@echo ""
	@echo "✅ Published to PyPI."
	@echo "   Install with: pip install hydra-suite"

# =============================================================================
# DOCUMENTATION
# =============================================================================

docs-install:
	"$(PYTHON_BIN)" -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['docs']))" > .docs-reqs.txt
	"$(PYTHON_BIN)" -m pip install -r .docs-reqs.txt -c constraints/base.txt && rm -f .docs-reqs.txt

install-dev:
	@echo "🔧 Installing dev & code-quality tools (pyproject [dev] extra, tested pins)..."
	"$(PYTHON_BIN)" -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['optional-dependencies']['dev']))" > .dev-reqs.txt
	"$(PYTHON_BIN)" -m pip install -r .dev-reqs.txt -c constraints/base.txt && rm -f .dev-reqs.txt
	@echo "⚠️  dep-graph also needs the graphviz dot binary: conda install -c conda-forge graphviz"

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --strict

docs-quality:
	python tools/doc_quality_check.py --baseline docs/doc-quality-baseline.json --min-module-doc 90 --min-symbol-doc 55 --min-typed-func 20

docs-check: docs-build docs-quality
	@echo "Checking docs terminology..."
	@set -e; \
	if command -v rg >/dev/null 2>&1; then \
		if rg -n "labeller|posekit-labeller" docs README.md mkdocs.yml; then \
			echo "Found non-canonical labeler spelling"; \
			exit 1; \
		fi; \
	else \
		if grep -RInE "labeller|posekit-labeller" docs README.md mkdocs.yml; then \
			echo "Found non-canonical labeler spelling"; \
			exit 1; \
		fi; \
	fi
	@echo "Docs checks passed."

techref-build:
	$(MAKE) -C technical-reference pdf

techref-clean:
	$(MAKE) -C technical-reference clean

# Pre-commit hooks
pre-commit-install:
	$(PRE_COMMIT) install
	@echo "Pre-commit hooks installed. They will run automatically on git commit."

pre-commit-autopep8:
	@echo "🧹 Running autopep8 pre-fix for common pycodestyle issues..."
	uvx autopep8 --in-place --recursive --select=E226,E225,E231 src/ tests/ tools/ legacy/
	@set -e; \
	for f in *.py; do \
		if [ -f "$$f" ]; then \
			uvx autopep8 --in-place --select=E226,E225,E231 "$$f"; \
		fi; \
	done

pre-commit-run:
	@$(MAKE) format
	@echo "🔎 Running pre-commit hooks (pass 1 — auto-fix)..."
	$(PRE_COMMIT) run --all-files || true
	@echo "adding unstaged changes after auto-fix..."
	git add -u
	@echo "🔎 Running pre-commit hooks (pass 2 — verify all pass)..."
	$(PRE_COMMIT) run --all-files
	@echo "✅ All pre-commit hooks passed. Ready to commit!"

pre-commit-update:
	$(PRE_COMMIT) autoupdate

# =============================================================================
# CODE QUALITY
# =============================================================================

# Format code: autopep8 whitespace fixes → black → isort
format:
	@echo "✨ Formatting code (autopep8 → black → isort)..."
	uvx autopep8 --in-place --recursive --select=E226,E225,E231 src/ tests/ tools/
	"$(PYTHON_BIN)" -m black src/ tests/ tools/
	"$(PYTHON_BIN)" -m isort src/ tests/ tools/
	@echo "✅ Format complete."

format-check:
	"$(PYTHON_BIN)" -m black --check src/ tests/ tools/
	"$(PYTHON_BIN)" -m isort --check-only src/ tests/ tools/
	@echo "Format check complete."

# Lint at moderate severity (default gate — catches real issues without noise)
# `lint-moderate` is an alias: CLAUDE.md's pre-PR checklist names it, and a
# missing target chained as `make lint-moderate && git commit` silently never
# reaches the commit.
lint-moderate: lint

lint:
	@echo "🔍 Linting (flake8 moderate)..."
	"$(PYTHON_BIN)" -m flake8 --config=.flake8.moderate src/ tests/ tools/
	@echo "✅ Lint complete."

# Auto-fix safe issues with ruff, then reformat
lint-fix:
	@echo "🛠️  Auto-fixing lint issues (ruff) then formatting..."
	@set +e; \
	uvx ruff check --fix --select F401,F541,F841 src/ tests/ tools/; \
	RUFF_EXIT=$$?; \
	set -e; \
	if [ $$RUFF_EXIT -ne 0 ]; then \
		echo "ℹ️  Ruff fixed what it could; remaining issues need manual edits."; \
	fi
	@$(MAKE) format
	@echo "✅ Auto-fix complete. Review with: git diff"

# Strict lint: all best-practice issues
lint-strict:
	@echo "🔍 Running strict linting (all issues)..."
	"$(PYTHON_BIN)" -m flake8 --config=.flake8.strict src/ tests/ tools/
	@echo "✅ Strict linting complete."

# Side-by-side comparison of all three strictness levels
lint-report:
	@echo "📊 Lint issue counts at all strictness levels:"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 LENIENT  — pre-commit / CI gate"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 MODERATE — make lint (recommended default)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.moderate src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 STRICT   — make lint-strict (best practices)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@flake8 --config=.flake8.strict src/ tests/ tools/ | wc -l | xargs -I {} echo "{} issues"
	@echo ""

# =============================================================================
# CODE HEALTH & DEPENDENCY AUDITING
# =============================================================================

# Find unused code (dead functions, classes, variables, imports)
dead-code:
	@echo "🔍 Scanning for dead / orphaned code (vulture, ≥80% confidence)..."
	@echo ""
	"$(PYTHON_BIN)" -m vulture src/hydra_suite --min-confidence 80
	@echo ""
	@echo "� Cross-checking with deadcode..."
	@echo ""
	deadcode src/hydra_suite
	@echo ""
	@echo "💡 To whitelist false positives (vulture):"
	@echo "   vulture src/hydra_suite --make-whitelist > vulture_whitelist.py"
	@echo "   vulture src/hydra_suite vulture_whitelist.py"
	@echo "💡 To auto-remove confirmed dead code: make dead-code-fix"

# Automatically remove dead code (caution: modifies source files — commit first!)
dead-code-fix:
	@echo "🗑️  Running deadcode --fix on src/hydra_suite ..."
	@echo "   ⚠️  This will MODIFY source files. Commit or stash changes first."
	deadcode src/hydra_suite --fix
	@echo "✅ Done. Review with: git diff"

# Generate visual dependency graph (renders to hydra_suite.svg in cwd)
dep-graph:
	@echo "🗺️  Generating dependency graph (pydeps) → hydra_suite.svg ..."
	@if ! command -v dot >/dev/null 2>&1; then \
		echo ""; \
		echo "ERROR: 'dot' (graphviz) not found on PATH."; \
		echo "Install it in your active conda environment:"; \
		echo "  conda install -c conda-forge graphviz"; \
		echo "Or update the environment and reinstall:"; \
		echo "  make env-update-mps  # (or env-update / env-update-cuda)"; \
		echo ""; \
		exit 1; \
	 fi
	pydeps src/hydra_suite \
		--max-bacon=4 \
		--cluster \
		--rankdir LR \
		--noshow \
		-o hydra_suite.svg
	@echo "✅ Graph written to hydra_suite.svg"
	@echo "   Open it in a browser or SVG viewer."

# Text-based module dependency list (no graphviz required)
dep-graph-text:
	@echo "🗺️  Module dependency map (pyreverse / pylint) ..."
	@mkdir -p .audit
	pyreverse -o dot -p hydra_suite src/hydra_suite -d .audit/ 2>/dev/null || true
	@if [ -f .audit/packages_hydra_suite.dot ]; then \
		echo ""; \
		echo "--- packages_hydra_suite.dot (raw DOT source) ---"; \
		cat .audit/packages_hydra_suite.dot; \
	else \
		echo "ℹ️  pyreverse produced no output – check pylint/graphviz installation."; \
		echo "   Falling back to pydeps text mode ..."; \
		pydeps src/hydra_suite --max-bacon=4 --noshow --nodot 2>/dev/null || true; \
	fi

# Static type checking with mypy
type-check:
	@echo "🔎 Running mypy static type check..."
	"$(PYTHON_BIN)" -m mypy src/hydra_suite --ignore-missing-imports --no-error-summary
	@echo "✅ mypy check complete."

# Full code-health audit: dead code + dep graph + type check + coverage
audit: dead-code dep-graph type-check test-cov
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Full audit complete."
	@echo "   Artifacts: hydra_suite.svg  htmlcov/index.html"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# =============================================================================
# HELP
# =============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════════════╗"
	@echo "║         HYDRA Suite - Development Commands                   ║"
	@echo "╚════════════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 QUICK START  (choose your platform, then follow printed instructions)"
	@echo "  make setup           - CPU / NumPy+Numba"
	@echo "  make setup-mps       - Apple Silicon (M1/M2/M3/M4)"
	@echo "  make setup-cuda      - NVIDIA GPU (CUDA)"
	@echo ""
	@echo "📦 INSTALL (after activating environment)"
	@echo "  make install[-mps|-cuda]          - Install runtime packages"
	@echo "  make install-dev                  - ⭐ Install dev & audit tools"
	@echo "  make docs-install                 - Install MkDocs dependencies"
	@echo ""
	@echo "🔄 ENVIRONMENT MAINTENANCE"
	@echo "  make env-create[-mps|-cuda] - Create conda environment"
	@echo "  make env-update[-mps|-cuda] - Update conda environment"
	@echo "  make env-remove[-mps|-cuda] - Remove conda environment"
	@echo ""
	@echo "🧪 TESTING"
	@echo "  make pytest          - Run all tests"
	@echo "  make test-cov        - Run tests with coverage report (terminal)"
	@echo "  make test-cov-html   - Run tests with HTML coverage (htmlcov/index.html)"
	@echo "  make clean           - Remove Python cache files"
	@echo ""
	@echo "✨ CODE QUALITY  (requires: make install-dev)"
	@echo "  make format          - ⭐ Format code: autopep8 → black → isort"
	@echo "  make format-check    - Check formatting without making changes"
	@echo "  make lint            - Lint at moderate severity (recommended gate)"
	@echo "  make lint-fix        - Auto-fix safe issues (ruff) then reformat"
	@echo "  make lint-strict     - Lint at maximum strictness"
	@echo "  make lint-report     - Side-by-side issue counts at all levels"
	@echo "  make pre-commit-install  - Install git pre-commit hooks"
	@echo "  make pre-commit-run      - Run pre-commit hooks on all files"
	@echo ""
	@echo "🩺 CODE HEALTH  (requires: make install-dev)"
	@echo "  make dead-code       - Find unused code (vulture)"
	@echo "  make dead-code-fix   - ⚠️  Auto-remove dead code in-place (commit first!)"
	@echo "  make dep-graph       - Visual SVG dependency graph → hydra_suite.svg"
	@echo "  make dep-graph-text  - Text module map (pyreverse, no graphviz needed)"
	@echo "  make type-check      - Static type checking (mypy)"
	@echo "  make audit           - Full sweep: dead-code + dep-graph + types + coverage"
	@echo ""
	@echo "📦 PACKAGING & PUBLISHING  (requires: make install-dev)"
	@echo "  make build           - Build wheel and sdist"
	@echo "  make publish-test    - Build + upload to Test PyPI"
	@echo "  make publish         - Build + upload to PyPI (real)"
	@echo ""
	@echo "📚 DOCUMENTATION"
	@echo "  make docs-serve      - Live preview at http://127.0.0.1:8000"
	@echo "  make docs-build      - Build (strict mode)"
	@echo "  make docs-check      - Build + quality metrics + terminology checks"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Platform notes: CPU=everywhere  MPS=Apple M-series  CUDA=NVIDIA"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
