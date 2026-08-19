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
#   RUNTIME   cpu|mps|cuda|onnx_cpu|onnx_cuda|tensorrt|gpu|gpu_fast|auto (default auto)
#             gpu/gpu_fast are the redesign's runtime_tier names (mapped to
#             cuda/tensorrt for the legacy per-stage fields by runner.py).
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
  cpu|mps|cuda|onnx_cpu|onnx_cuda|tensorrt|gpu|gpu_fast|config) ;;
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
    "worm_bgsub_scaled|$FX/clips/worm_bgsub.mp4|$FX/configs/worm_bgsub_scaled.json|"
    "ant_cnn_identity|$FX/clips/ant_cnn_identity.mp4|$FX/configs/ant_cnn_identity.json|$FX/ooceraea_biroi.json"
    "fly_obb|$FX/clips/fly_obb.mp4|$FX/configs/fly_obb.json|"
  )
  # ON-path clips: configs that deliberately turn a feature ON, so their
  # legacy-vs-new EQUIVALENCE line is EXPECTED to differ and means nothing.
  # They are excluded from the default matrix for exactly that reason -- a
  # red line here would poison an otherwise-green gate. What they DO prove is
  # what the default matrix cannot: that the feature runs on real video and
  # is deterministic. Run with ONPATH=1 and read only the DETERMINISM line
  # (new_a vs new_b), which must be clean, plus the identity section.
  #   ONPATH=1 MAIN_SRC=$WT_SRC ... bash run_matrix.sh ant_cnn_identity_marked
  # Setting MAIN_SRC=WT_SRC makes all three runs the new code, so every
  # printed comparison becomes a determinism check.
  if [ "${ONPATH:-0}" = "1" ]; then
    VIDEOS+=(
      "ant_cnn_identity_marked|$FX/clips/ant_cnn_identity.mp4|$FX/configs/ant_cnn_identity_marked.json|$FX/ooceraea_biroi.json"
    )
  fi
else
  VIDEOS=(
    "emi_short|$DATA/ant/emi_short.mp4|$DATA/ant/emi_short_config.json|"
    "ant2|$DATA/ant2/000001_cropped_roi.mp4|$DATA/ant2/000001_cropped_roi_config.json|$FX/ooceraea_biroi.json"
  )
fi

# Optionally restrict to specific clips, so you don't rerun the whole matrix.
# Pass names as arguments or via ONLY="a b" (space- or comma-separated):
#   bash tools/equivalence/run_matrix.sh ant_pose_headtail worm_bgsub
# worm_bgsub_scaled is the only clip exercising RESIZE_FACTOR < 1.0 (Scale=0.5);
# bg-sub is the sole method that honors it, so it is the sole scaled fixture.
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

# Clips whose comparison could not be trusted. A harness that prints a green
# verdict for a run that crashed is worse than one that prints nothing, so every
# untrustworthy outcome lands here and makes the whole matrix exit non-zero.
FAILED_CLIPS=()

note_failure() {  # clip reason
  FAILED_CLIPS+=("$1: $2")
}

# A CSV with only a header is NOT a result. Tracking writes the header before it
# runs, so a crashed run leaves a well-formed, existent, empty file behind --
# which every existence check in this script used to accept.
has_rows() {  # path
  [ -f "$1" ] || return 1
  [ "$(wc -l < "$1")" -gt 1 ] || return 1
}

cmp() {  # a b title clip [extra compare.py args...]
  echo "--- $3 ---"
  # Not every clip config produces every CSV kind: a clip with streaming
  # individual analysis (e.g. ant_pose_headtail) writes only *_final.csv and
  # *_final_with_individual.csv, never an intermediate *_forward.csv. Absent on
  # BOTH sides is that config property and is not a failure. Absent on ONE side
  # means the trees disagree about what they produced, which is.
  if ! [ -f "$1" ] && ! [ -f "$2" ]; then
    echo "  (not produced by either tree for this clip -- config does not emit it)"
    return
  fi
  if ! [ -f "$1" ] || ! [ -f "$2" ]; then
    echo "  ❌ MISSING ON ONE SIDE ONLY: exists=$([ -f "$1" ] && echo a || echo b), missing=$([ -f "$1" ] && echo b || echo a)"
    note_failure "${4:-?}" "$3 -- CSV produced by one tree but not the other"
    return
  fi
  if ! has_rows "$1" || ! has_rows "$2"; then
    echo "  ❌ EMPTY CSV (header only): legacy=$(wc -l < "$1") lines, new=$(wc -l < "$2") lines"
    echo "     A crashed run leaves a header-only CSV; comparing two of them"
    echo "     satisfies every criterion vacuously. Check the tracking log."
    note_failure "${4:-?}" "$3 -- empty CSV (header only)"
    return
  fi
  python "$WT/tools/equivalence/compare.py" "$1" "$2" "${@:5}"
  rc=$?
  # compare.py: 0 = equivalent, 1 = real differences, 2 = no data
  if [ "$rc" = "2" ]; then
    note_failure "${4:-?}" "$3 -- compare.py reported NO DATA"
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
  run "$MAIN_SRC" "$base/legacy" "$config" "$video" "legacy" "$skeleton" \
    || { echo "!! legacy run FAILED"; note_failure "$name" "legacy run exited non-zero"; }
  run "$WT_SRC"   "$base/new_a"  "$config" "$video" "new_a"  "$skeleton" \
    || { echo "!! new_a run FAILED";  note_failure "$name" "new_a run exited non-zero"; }
  run "$WT_SRC"   "$base/new_b"  "$config" "$video" "new_b"  "$skeleton" \
    || { echo "!! new_b run FAILED";  note_failure "$name" "new_b run exited non-zero"; }

  stem=$(basename "$video"); stem=${stem%.*}
  for kind in forward final; do
    echo; echo ">>> $name : $kind"
    cmp "$base/new_a/${stem}_tracking_${kind}.csv" \
        "$base/new_b/${stem}_tracking_${kind}.csv" \
        "DETERMINISM  new_a vs new_b" "$name"
    cmp "$base/legacy/${stem}_tracking_${kind}.csv" \
        "$base/new_a/${stem}_tracking_${kind}.csv" \
        "EQUIVALENCE  legacy vs new_a" "$name"
  done

  # The rich per-individual CSV is the ONLY export carrying the identity
  # columns (IdentityEvidence*/IdentityFinal*/UniqueIdentityKey), pose, and the
  # per-classifier CNN columns. Without it this matrix proves geometry and
  # tracking were not perturbed and nothing at all about identity. Compared
  # with --strict-columns so those (mostly non-numeric) columns are gated
  # rather than merely printed. Clips with no individual-analysis stage
  # produce no such file; `cmp` reports "not produced by either tree" for
  # that -- absence on BOTH sides is a config property, absence on ONE side
  # is still a failure.
  echo; echo ">>> $name : final_with_individual (identity columns)"
  cmp "$base/new_a/${stem}_tracking_final_with_individual.csv" \
      "$base/new_b/${stem}_tracking_final_with_individual.csv" \
      "DETERMINISM  new_a vs new_b" "$name" --strict-columns
  cmp "$base/legacy/${stem}_tracking_final_with_individual.csv" \
      "$base/new_a/${stem}_tracking_final_with_individual.csv" \
      "EQUIVALENCE  legacy vs new_a" "$name" --strict-columns

  echo; echo ">>> $name : performance"
  perfcmp "$base/legacy/meta.json" "$base/new_a/meta.json"
done
echo; echo "### done. outputs under $OUT/$RUNTIME/"
echo
if [ ${#FAILED_CLIPS[@]} -gt 0 ]; then
  echo "############################################################"
  echo "### ${#FAILED_CLIPS[@]} UNTRUSTWORTHY RESULT(S) -- do NOT read this matrix as a pass:"
  for f in "${FAILED_CLIPS[@]}"; do echo "###   - $f"; done
  echo "############################################################"
  exit 1
fi
echo "### all clips produced comparable output."
