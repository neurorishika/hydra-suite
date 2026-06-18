#!/usr/bin/env bash
# One-shot equivalence matrix: legacy (main) vs new (worktree) across videos,
# on whatever device this machine has (auto-detected). For each video it runs
# legacy once and the new pipeline twice, then prints:
#   - determinism baseline: new_a vs new_b  (the noise floor of one pipeline)
#   - equivalence:          legacy vs new_a (must be within that floor)
#
# Configure via env vars (defaults assume this repo layout):
#   REPO      repo root (has src/ on the main branch)
#   WT        worktree path (the redesign branch)
#   MAIN_SRC  source tree for the LEGACY pipeline   (default $REPO/src)
#   WT_SRC    source tree for the NEW pipeline       (default $WT/src)
#   DATA      MultiTrackerData root
#   OUT       output root (default /tmp/equiv)
#   RUNTIME   cpu|mps|cuda|onnx_cpu|onnx_cuda|tensorrt|auto (default auto)
#
# Edit the VIDEOS list below to add targets.  Nothing writes into DATA: each run
# symlinks the video into its own output dir.
set -uo pipefail

REPO=${REPO:-/Users/neurorishika/Projects/Rockefeller/Kronauer/multi-animal-tracker}
WT=${WT:-$REPO/.worktrees/inference-pipeline-redesign}
MAIN_SRC=${MAIN_SRC:-$REPO/src}
WT_SRC=${WT_SRC:-$WT/src}
DATA=${DATA:-/Users/neurorishika/Projects/Rockefeller/RutaKronauer/MultiTrackerData}
OUT=${OUT:-/tmp/equiv}
RUNTIME=${RUNTIME:-auto}

# Conda/torch builds often link libomp twice; without this, OpenMP calls abort()
# (the "OMP Error #15" you saw), which killed the device-autodetect subshell and
# every runner. Export so all child python processes inherit it.
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}

if [ "$RUNTIME" = "auto" ]; then
  RUNTIME=$(python - <<'PY'
import torch
b = getattr(torch.backends, "mps", None)
print("cuda" if torch.cuda.is_available()
      else ("mps" if (b and torch.backends.mps.is_available()) else "cpu"))
PY
)
fi
case "$RUNTIME" in
  cpu|mps|cuda|onnx_cpu|onnx_cuda|tensorrt|config) ;;
  *)
    echo "!! Could not determine a valid runtime (got '$RUNTIME')." >&2
    echo "   Set it explicitly, e.g.  RUNTIME=mps bash tools/equivalence/run_matrix.sh" >&2
    exit 2 ;;
esac
echo "### runtime = $RUNTIME"
echo "### legacy src = $MAIN_SRC"
echo "### new    src = $WT_SRC"

# name | video | config
VIDEOS=(
  "emi_short|$DATA/ant/emi_short.mp4|$DATA/ant/emi_short_config.json"
  "ant2|$DATA/ant2/000001_cropped_roi.mp4|$DATA/ant2/000001_cropped_roi_config.json"
)

run() {  # src outdir config video label
  PYTHONPATH="$1" python "$WT/tools/equivalence/runner.py" \
    --orig-config "$3" --video "$4" --outdir "$2" --runtime "$RUNTIME" --label "$5"
}

cmp() {  # a b title
  echo "--- $3 ---"
  if [ -f "$1" ] && [ -f "$2" ]; then
    python "$WT/tools/equivalence/compare.py" "$1" "$2" || true
  else
    echo "  (missing CSV: $1 or $2)"
  fi
}

for entry in "${VIDEOS[@]}"; do
  IFS='|' read -r name video config <<< "$entry"
  base="$OUT/$RUNTIME/$name"
  echo; echo "============================================================"
  echo "=== $name  ($RUNTIME)"
  echo "============================================================"
  run "$MAIN_SRC" "$base/legacy" "$config" "$video" "legacy" || echo "!! legacy run failed"
  run "$WT_SRC"   "$base/new_a"  "$config" "$video" "new_a"  || echo "!! new_a run failed"
  run "$WT_SRC"   "$base/new_b"  "$config" "$video" "new_b"  || echo "!! new_b run failed"

  stem=$(basename "$video"); stem=${stem%.*}
  for kind in forward final; do
    echo; echo ">>> $name : $kind"
    cmp "$base/new_a/${stem}_tracking_${kind}.csv" \
        "$base/new_b/${stem}_tracking_${kind}.csv" \
        "DETERMINISM  new_a vs new_b"
    cmp "$base/legacy/${stem}_tracking_${kind}.csv" \
        "$base/new_a/${stem}_tracking_${kind}.csv" \
        "EQUIVALENCE  legacy vs new_a"
  done
done
echo; echo "### done. outputs under $OUT/$RUNTIME/"
