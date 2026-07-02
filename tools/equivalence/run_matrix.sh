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

# Target set. FIXTURES=1 (default) uses the committed short clips under
# fixtures/ (run fetch_fixtures.sh first). FIXTURES=0 uses the full local videos
# under $DATA. Format: name | video | config | skeleton(optional)
FIXTURES=${FIXTURES:-1}
FX="$WT/tools/equivalence/fixtures"
if [ "$FIXTURES" = "1" ]; then
  VIDEOS=(
    "emi_obb_identity|$FX/clips/emi_obb_identity.mp4|$FX/configs/emi_obb_identity.json|"
    "ant_pose_headtail|$FX/clips/ant_pose_headtail.mp4|$FX/configs/ant_pose_headtail.json|$FX/ooceraea_biroi.json"
    "ant_obb_sleap|$FX/clips/ant_obb_sleap.mp4|$FX/configs/ant_obb_sleap.json|$FX/ooceraea_biroi.json"
    "ant_obb_sequential|$FX/clips/ant_obb_sleap.mp4|$FX/configs/ant_obb_sequential.json|$FX/ooceraea_biroi.json"
    "worm_bgsub|$FX/clips/worm_bgsub.mp4|$FX/configs/worm_bgsub.json|"
    "ant_cnn_identity|$FX/clips/ant_cnn_identity.mp4|$FX/configs/ant_cnn_identity.json|$FX/ooceraea_biroi.json"
    "fly_obb|$FX/clips/fly_obb.mp4|$FX/configs/fly_obb.json|"
  )
else
  VIDEOS=(
    "emi_short|$DATA/ant/emi_short.mp4|$DATA/ant/emi_short_config.json|"
    "ant2|$DATA/ant2/000001_cropped_roi.mp4|$DATA/ant2/000001_cropped_roi_config.json|$FX/ooceraea_biroi.json"
  )
fi

# Optionally restrict to specific clips, so you don't rerun the whole matrix.
# Pass names as arguments or via ONLY="a b" (space- or comma-separated):
#   bash tools/equivalence/run_matrix.sh ant_pose_headtail worm_bgsub
#   ONLY=ant_pose_headtail bash tools/equivalence/run_matrix.sh
_only="${ONLY:-$*}"
_only="${_only//,/ }"  # allow comma-separated too
if [ -n "$_only" ]; then
  _filtered=()
  for entry in "${VIDEOS[@]}"; do
    for want in $_only; do
      if [ "${entry%%|*}" = "$want" ]; then
        _filtered+=("$entry")
        break
      fi
    done
  done
  if [ "${#_filtered[@]}" -eq 0 ]; then
    echo "!! No clips matched: $_only" >&2
    printf '   Available:' >&2
    for e in "${VIDEOS[@]}"; do printf ' %s' "${e%%|*}" >&2; done
    echo >&2
    exit 2
  fi
  VIDEOS=("${_filtered[@]}")
  echo "### subset: $_only"
fi

run() {  # src outdir config video label skeleton
  local skel_arg=()
  [ -n "${6:-}" ] && skel_arg=(--skeleton "$6")
  PYTHONPATH="$1" python "$WT/tools/equivalence/runner.py" \
    --orig-config "$3" --video "$4" --outdir "$2" --runtime "$RUNTIME" --label "$5" \
    ${skel_arg[@]+"${skel_arg[@]}"}
}

cmp() {  # a b title
  echo "--- $3 ---"
  if [ -f "$1" ] && [ -f "$2" ]; then
    python "$WT/tools/equivalence/compare.py" "$1" "$2" || true
  else
    echo "  (missing CSV: $1 or $2)"
  fi
}

# Performance gate: the new pipeline must not be meaningfully slower than legacy.
# Compares wall-clock/fps from each run's meta.json. PERF_TOLERANCE is the max
# allowed new/legacy time ratio (default 1.25 = new may be up to 25% slower).
PERF_TOLERANCE=${PERF_TOLERANCE:-1.25}
perfcmp() {  # legacy_meta new_meta
  echo "--- PERFORMANCE  legacy vs new_a (tolerance ${PERF_TOLERANCE}x) ---"
  if [ ! -f "$1" ] || [ ! -f "$2" ]; then
    echo "  (missing meta.json)"; return
  fi
  PERF_TOLERANCE="$PERF_TOLERANCE" python - "$1" "$2" <<'PY'
import json, os, sys
leg = json.load(open(sys.argv[1])); new = json.load(open(sys.argv[2]))
lt, nt = leg.get("tracking_seconds"), new.get("tracking_seconds")
lf, nf = leg.get("fps"), new.get("fps")
tol = float(os.environ.get("PERF_TOLERANCE", "1.25"))
print(f"  legacy: {lt}s ({lf} fps)   new: {nt}s ({nf} fps)")
if not lt or not nt:
    print("  (no timing recorded)"); raise SystemExit(0)
ratio = nt / lt
verdict = "EQUIVALENT ✅" if ratio <= tol else "SLOWER ❌"
print(f"  new/legacy time ratio = {ratio:.2f}x  ->  PERFORMANCE: {verdict}")
PY
}

for entry in "${VIDEOS[@]}"; do
  IFS='|' read -r name video config skeleton <<< "$entry"
  base="$OUT/$RUNTIME/$name"
  echo; echo "============================================================"
  echo "=== $name  ($RUNTIME)"
  echo "============================================================"
  run "$MAIN_SRC" "$base/legacy" "$config" "$video" "legacy" "$skeleton" || echo "!! legacy run failed"
  run "$WT_SRC"   "$base/new_a"  "$config" "$video" "new_a"  "$skeleton" || echo "!! new_a run failed"
  run "$WT_SRC"   "$base/new_b"  "$config" "$video" "new_b"  "$skeleton" || echo "!! new_b run failed"

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

  echo; echo ">>> $name : performance"
  perfcmp "$base/legacy/meta.json" "$base/new_a/meta.json"
done
echo; echo "### done. outputs under $OUT/$RUNTIME/"
