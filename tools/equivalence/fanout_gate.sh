#!/usr/bin/env bash
# Fan-out vs sequential byte-identity gate.
#
# Proves that `trackerkit track --video-list ... --jobs N [--gpus ...]` (one
# child process per slot) produces byte-identical CSVs to the in-process
# sequential path for the same batch. Two staging dirs (seq/ and par/) hold
# symlinked clips plus per-video sidecar configs, so the two runs never share
# an output path or a detection cache.
#
# Run with the platform conda env ACTIVE (hydra-mps here, hydra-cuda on mehek);
# pose/SLEAP clips silently produce header-only CSVs without it, and comparing
# two empty files passes vacuously -- hence the row-count guard below.
#
#   OUT=/tmp/fanout_gate bash tools/equivalence/fanout_gate.sh \
#       fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity
#
# Env knobs:
#   OUT         output root (default /tmp/fanout_gate)
#   FIXTURES    fixtures dir (default <repo>/tools/equivalence/fixtures)
#   CLIPS_DIR   clips dir    (default $FIXTURES/clips; clips are gitignored and
#               fetched per machine, so a worktree may need this pointed at the
#               main checkout)
#   RUNTIME     cpu|mps|cuda|gpu|gpu_fast|... (default: auto-detect, as run_matrix)
#   JOBS        --jobs value for the fan-out run (default 2)
#   EXTRA       extra flags for the fan-out run, e.g. "--threads-per-job 4" or "--gpus 0"
#   INHERIT     1 = keystone-inheritance leg: ONLY the first clip gets a sidecar
#               (the rest inherit it through plan_batch_jobs, which is the default
#               GUI batch flow and the branch the plain leg never executes), and
#               that sidecar has video_output_enabled=true so the per-video
#               annotated overlay -- the one output that IS taken from the
#               inherited config -- is actually exercised.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

# The src tree under test is the one this script ships with -- not $PWD, which
# may be the main checkout when a worktree's gate is being run.
export PYTHONPATH="$ROOT/src"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

FIXTURES="${FIXTURES:-$ROOT/tools/equivalence/fixtures}"
CLIPS_DIR="${CLIPS_DIR:-$FIXTURES/clips}"
OUT="${OUT:-/tmp/fanout_gate}"
JOBS="${JOBS:-2}"
INHERIT="${INHERIT:-0}"
EXTRA="${EXTRA:-}"
RUNTIME="${RUNTIME:-auto}"

if [ "$RUNTIME" = "auto" ]; then
  RUNTIME=$(python - <<'PY'
import torch
b = getattr(torch.backends, "mps", None)
print("cuda" if torch.cuda.is_available()
      else ("mps" if (b and torch.backends.mps.is_available()) else "cpu"))
PY
)
fi

# name|video|config|skeleton -- mirrors run_matrix.sh's fixture table so a clip
# is configured here exactly as the standard matrix configures it.
# NOTE: two clips that share one .mp4 (ant_obb_sequential/ant_obb_sleap,
# worm_bgsub_scaled/worm_bgsub, ant_cnn_identity_relink/ant_cnn_identity) can
# never appear in the SAME batch: plan_batch_jobs._reject_collisions compares
# os.path.realpath and refuses a batch that lists one video twice. Each clip is
# staged under its own name, but the symlinks resolve to the same file.
ALL_ENTRIES=(
  "fly_obb|$CLIPS_DIR/fly_obb.mp4|$FIXTURES/configs/fly_obb.json|"
  "worm_bgsub|$CLIPS_DIR/worm_bgsub.mp4|$FIXTURES/configs/worm_bgsub.json|"
  "ant_obb_sleap|$CLIPS_DIR/ant_obb_sleap.mp4|$FIXTURES/configs/ant_obb_sleap.json|$FIXTURES/ooceraea_biroi.json"
  "ant_pose_headtail|$CLIPS_DIR/ant_pose_headtail.mp4|$FIXTURES/configs/ant_pose_headtail.json|$FIXTURES/ooceraea_biroi.json"
  "ant_cnn_identity|$CLIPS_DIR/ant_cnn_identity.mp4|$FIXTURES/configs/ant_cnn_identity.json|$FIXTURES/ooceraea_biroi.json"
)

WANT=("$@")
if [ "${#WANT[@]}" -eq 0 ]; then
  WANT=(fly_obb worm_bgsub ant_obb_sleap ant_cnn_identity)
fi

ENTRIES=()
for want in "${WANT[@]}"; do
  found=""
  for entry in "${ALL_ENTRIES[@]}"; do
    if [ "${entry%%|*}" = "$want" ]; then ENTRIES+=("$entry"); found=1; break; fi
  done
  if [ -z "$found" ]; then
    echo "!! unknown clip: $want" >&2
    printf '   available:' >&2
    for e in "${ALL_ENTRIES[@]}"; do printf ' %s' "${e%%|*}" >&2; done
    echo >&2
    exit 2
  fi
done

echo "### fan-out gate"
echo "### repo root = $ROOT"
echo "### runtime   = $RUNTIME"
echo "### clips     = $CLIPS_DIR"
echo "### out       = $OUT"
echo "### jobs      = $JOBS   extra = '${EXTRA:-<none>}'"
echo "### inherit   = $INHERIT (1 = keystone-only sidecar + annotated video)"
echo "### clips under test: ${WANT[*]}"

# A stale __pycache__ (especially a numba @jit(cache=True) entry left by another
# worktree at the same path) can silently swap an algorithm under us.
find "$ROOT/src" -name __pycache__ -exec rm -rf {} + 2>/dev/null

echo "### thread-cap env as inherited by children:"
env | grep -E '^(OMP|MKL|OPENBLAS|NUMEXPR|VECLIB)_[A-Z_]*THREADS=' || echo "    (none set)"

hydra_file=$(python -c 'import hydra_suite; print(hydra_suite.__file__)')
echo "### hydra_suite = $hydra_file"
case "$hydra_file" in
  "$ROOT/src/"*) ;;
  *) echo "!! hydra_suite resolves OUTSIDE $ROOT/src -- the wrong tree would be gated." >&2
     exit 2 ;;
esac

rm -rf "$OUT"
mkdir -p "$OUT/seq" "$OUT/par"

# Stage through runner.py's own build_config so the sidecar is exactly what the
# equivalence harness would hand this clip: blanked paths filled in, side
# outputs disabled, and the runtime tier injected (the fixture configs carry no
# runtime_tier, and loading one without it raises the migration error).
python - "$OUT" "$RUNTIME" "$HERE" "$INHERIT" "${ENTRIES[@]}" <<'PY'
import json
import shutil
import sys
from pathlib import Path

out = Path(sys.argv[1])
runtime = sys.argv[2]
sys.path.insert(0, sys.argv[3])
inherit = sys.argv[4] == "1"
import runner  # noqa: E402

for index, entry in enumerate(sys.argv[5:]):
    name, video, config, skeleton = entry.split("|")
    source = Path(video)
    if not source.is_file():
        raise SystemExit(f"!! clip not found: {source}")
    for mode in ("seq", "par"):
        stage = out / mode
        stage.mkdir(parents=True, exist_ok=True)
        link = stage / f"{name}.mp4"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source.resolve())
        cfg = runner.build_config(
            config, link, stage, runtime, skeleton=(skeleton or None)
        )
        if inherit and index > 0:
            # No sidecar: this video must inherit the keystone's config through
            # the planner. That branch is what the plain leg never executes.
            Path(cfg).unlink()
            print(f"    staged {mode}/{name}.mp4 (no sidecar: inherits keystone)")
            continue
        dest = stage / f"{name}_config.json"
        shutil.move(str(cfg), str(dest))
        if inherit:
            # runner.build_config force-disables every side output; re-enable the
            # annotated video on the KEYSTONE so the inherited-config path has a
            # per-video output to get wrong.
            data = json.loads(dest.read_text(encoding="utf-8"))
            data["video_output_enabled"] = True
            data["video_output_path"] = str(stage / f"{name}_tracking.mp4")
            dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"    staged {mode}/{name}.mp4 + {name}_config.json (KEYSTONE,"
                  " video_output_enabled=true)")
        else:
            print(f"    staged {mode}/{name}.mp4 + {name}_config.json")
PY
if [ $? -ne 0 ]; then echo "!! staging failed" >&2; exit 2; fi

list_seq="$OUT/seq/batch.txt"; list_par="$OUT/par/batch.txt"
: > "$list_seq"; : > "$list_par"
for entry in "${ENTRIES[@]}"; do
  clip="${entry%%|*}"
  echo "$OUT/seq/$clip.mp4" >> "$list_seq"
  echo "$OUT/par/$clip.mp4" >> "$list_par"
done

echo
echo "== sequential (in-process, no fan-out) =="
python -m hydra_suite.trackerkit.app track --video-list "$list_seq" > "$OUT/seq.log" 2>&1
seq_rc=$?
tail -3 "$OUT/seq.log"
echo "   rc=$seq_rc  log=$OUT/seq.log"

echo
echo "== fan-out (--jobs $JOBS ${EXTRA:-}) =="
# shellcheck disable=SC2086
python -m hydra_suite.trackerkit.app track --video-list "$list_par" --jobs "$JOBS" $EXTRA \
  > "$OUT/par.log" 2>&1
par_rc=$?
tail -12 "$OUT/par.log"
echo "   rc=$par_rc  log=$OUT/par.log"

status=0
[ "$seq_rc" -eq 0 ] || { echo "❌ sequential run exited $seq_rc"; status=1; }
[ "$par_rc" -eq 0 ] || { echo "❌ fan-out run exited $par_rc"; status=1; }

echo
echo "== fan-out actually engaged? =="
job_lines=$(grep -c '\[job ' "$OUT/par.log" 2>/dev/null || echo 0)
if [ "$job_lines" -gt 0 ]; then
  echo "✅ $job_lines child [job N] lines in par.log (children really ran)"
else
  echo "❌ no [job N] lines in par.log -- fan-out never engaged, so this gate is vacuous"
  status=1
fi
grep -E '^[0-9]+/[0-9]+ videos succeeded' "$OUT/par.log" || echo "   (no summary line)"
echo "-- child log headers (GPU pin + command) --"
for entry in "${ENTRIES[@]}"; do
  clip="${entry%%|*}"
  for log in "$OUT/par/${clip}_logs/"*_fanout_*.log; do
    [ -f "$log" ] || continue
    echo "   $clip: $(head -1 "$log")"
    echo "      $(grep -m1 '^# gpu=' "$log")"
  done
done

echo
echo "== byte-identity: sequential vs fan-out =="
for entry in "${ENTRIES[@]}"; do
  clip="${entry%%|*}"
  a_files=$(cd "$OUT/seq" && ls ${clip}*tracking*.csv 2>/dev/null | sort | tr '\n' ' ')
  b_files=$(cd "$OUT/par" && ls ${clip}*tracking*.csv 2>/dev/null | sort | tr '\n' ' ')
  if [ -z "$a_files" ]; then
    echo "❌ $clip: sequential produced NO tracking CSV at all"
    status=1
    continue
  fi
  if [ "$a_files" != "$b_files" ]; then
    echo "❌ $clip: different CSV sets -- seq='$a_files' par='$b_files'"
    status=1
    continue
  fi
  # A run that crashed still leaves the header it wrote up front, and two
  # header-only files compare equal. Require the forward + final pair and rows.
  for required in "${clip}_tracking_forward.csv" "${clip}_tracking_final.csv"; do
    case " $a_files " in
      *" $required "*) ;;
      *) echo "❌ $clip: expected $required was not produced"; status=1 ;;
    esac
  done
  for f in $a_files; do
    a="$OUT/seq/$f"; b="$OUT/par/$f"
    rows_a=$(wc -l < "$a" | tr -d ' ')
    rows_b=$(wc -l < "$b" | tr -d ' ')
    if [ "$rows_a" -le 1 ] || [ "$rows_b" -le 1 ]; then
      echo "❌ $f: header-only CSV (seq=$rows_a rows, par=$rows_b rows) -- empty run, not a pass"
      status=1
      continue
    fi
    if cmp -s "$a" "$b"; then
      echo "✅ $f byte-identical (seq=$rows_a rows, par=$rows_b rows)"
    else
      echo "❌ $f DIFFERS (seq=$rows_a rows, par=$rows_b rows)"
      status=1
    fi
  done
done

if [ "$INHERIT" = "1" ]; then
  echo
  echo "== side outputs: every clip must render its OWN annotated video =="
  # Pre-fix, an inheriting video kept the KEYSTONE's absolute video_output_path,
  # so all N children rendered into one file: the keystone's mp4 was written N
  # times (concurrently, in the fan-out) and no other clip had one at all.
  for mode in seq par; do
    for entry in "${ENTRIES[@]}"; do
      clip="${entry%%|*}"
      mp4="$OUT/$mode/${clip}_tracking.mp4"
      if [ -s "$mp4" ]; then
        echo "✅ $mode/${clip}_tracking.mp4 ($(wc -c < "$mp4" | tr -d ' ') bytes)"
      else
        echo "❌ $mode/${clip}_tracking.mp4 missing or empty -- no overlay of its own"
        status=1
      fi
    done
    produced=$(ls "$OUT/$mode/"*_tracking.mp4 2>/dev/null | wc -l | tr -d ' ')
    if [ "$produced" -eq "${#ENTRIES[@]}" ]; then
      echo "✅ $mode: ${#ENTRIES[@]} clips -> $produced distinct annotated videos"
    else
      echo "❌ $mode: ${#ENTRIES[@]} clips -> $produced annotated videos (paths collided)"
      ls -l "$OUT/$mode/"*.mp4 2>/dev/null || true
      status=1
    fi
  done
fi

echo
if [ "$status" -eq 0 ]; then
  echo "### GATE PASSED -- fan-out output is byte-identical to sequential."
else
  echo "### GATE FAILED -- see the ❌ lines above. Outputs kept under $OUT."
fi
exit $status
