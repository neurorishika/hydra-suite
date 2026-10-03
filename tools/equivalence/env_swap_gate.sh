#!/usr/bin/env bash
# Env-swap equivalence gate: same src tree, two Python ENVIRONMENTS.
#
# run_matrix.sh holds the env fixed and swaps the src tree. Install changes need
# the opposite: hold the src fixed and swap the interpreter (old conda env vs an
# env built by install.py), then prove the tracking CSVs are byte-identical.
#
# Usage:
#   OLD_PY=/path/old/bin/python NEW_PY=/path/new/bin/python \
#   SRC=$PWD/src OUT=/tmp/envswap RUNTIME=mps \
#   bash tools/equivalence/env_swap_gate.sh fly_obb worm_bgsub
#
# Either side may be skipped with SKIP_OLD=1 / SKIP_NEW=1 to reuse a previous run.
# Exits non-zero if any clip is missing, empty, or not equivalent. "Equivalent"
# = byte-identical, or differing only by BLAS float rounding in the Kalman
# diagnostic columns (see env_swap_compare.py).
set -euo pipefail

: "${OLD_PY:?set OLD_PY}" "${NEW_PY:?set NEW_PY}" "${SRC:?set SRC}" "${OUT:?set OUT}"
RUNTIME="${RUNTIME:-mps}"
WT="${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FX="${FX:-$WT/tools/equivalence/fixtures}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"

CLIPS=("$@")
[ ${#CLIPS[@]} -eq 0 ] && CLIPS=(fly_obb worm_bgsub)

clip_spec() {  # name -> "video|config|skeleton"
  case "$1" in
    fly_obb) echo "$FX/clips/fly_obb.mp4|$FX/configs/fly_obb.json|" ;;
    worm_bgsub) echo "$FX/clips/worm_bgsub.mp4|$FX/configs/worm_bgsub.json|" ;;
    ant_obb_sleap) echo "$FX/clips/ant_obb_sleap.mp4|$FX/configs/ant_obb_sleap.json|$FX/ooceraea_biroi.json" ;;
    *) echo "unknown clip $1" >&2; return 1 ;;
  esac
}

run_side() {  # python label clip
  local spec video config skel skel_arg=()
  spec="$(clip_spec "$3")"
  IFS='|' read -r video config skel <<<"$spec"
  [ -n "$skel" ] && skel_arg=(--skeleton "$skel")
  rm -rf "$OUT/$2/$3"
  PYTHONPATH="$SRC" "$1" "$WT/tools/equivalence/runner.py" \
    --orig-config "$config" --video "$video" --outdir "$OUT/$2/$3" \
    --runtime "$RUNTIME" --label "$2" ${skel_arg[@]+"${skel_arg[@]}"}
}

FAILED=()
for clip in "${CLIPS[@]}"; do
  [ "${SKIP_OLD:-0}" = 1 ] || run_side "$OLD_PY" old "$clip"
  [ "${SKIP_NEW:-0}" = 1 ] || run_side "$NEW_PY" new "$clip"
  for kind in forward final; do
    a=$(ls "$OUT/old/$clip"/*_tracking_"$kind".csv 2>/dev/null | head -1 || true)
    b=$(ls "$OUT/new/$clip"/*_tracking_"$kind".csv 2>/dev/null | head -1 || true)
    if [ -z "$a" ] && [ -z "$b" ]; then
      echo "[$clip/$kind] not produced by either env"; continue
    fi
    if [ -z "$a" ] || [ -z "$b" ] || [ "$(wc -l <"$a")" -le 1 ] || [ "$(wc -l <"$b")" -le 1 ]; then
      FAILED+=("$clip/$kind: missing or empty CSV"); continue
    fi
    if verdict=$(python3 "$WT/tools/equivalence/env_swap_compare.py" "$a" "$b"); then
      echo "[$clip/$kind] $verdict"
    else
      echo "[$clip/$kind] $verdict"
      FAILED+=("$clip/$kind: $verdict")
    fi
  done
done

if [ ${#FAILED[@]} -gt 0 ]; then
  printf 'FAIL: %s\n' "${FAILED[@]}"; exit 1
fi
echo "ALL EQUIVALENT (byte-identical or BLAS floor only)"
