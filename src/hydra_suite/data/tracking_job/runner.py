"""The generated run.sh: the job's executable contract."""

RUN_SH = """#!/usr/bin/env bash
set -euo pipefail
JOB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The HOST's config dir, captured BEFORE we override it: the shared-root mount
# table is a property of this machine, not of the job, and must never be read
# from the job snapshot. Empty means "the platformdirs default".
export HYDRA_HOST_CONFIG_DIR="${HYDRA_CONFIG_DIR:-}"

# The job supplies models and config; HYDRA_DATA_DIR is deliberately NOT set so
# engine artifacts and calibration profiles stay host-scoped.
export HYDRA_MODELS_DIR="$JOB/models"
export HYDRA_CONFIG_DIR="$JOB/config"
export KMP_DUPLICATE_LIB_OK=TRUE

if [ "${HYDRA_JOB_HOST_ADVANCED_CONFIG:-0}" = "1" ]; then
  # ****************************************************************
  # LOUD WARNING (minor fix -- this switch is a silent-divergence risk and
  # belongs documented HERE, at the escape hatch itself, not only in prose
  # elsewhere in this plan): HYDRA_JOB_HOST_ADVANCED_CONFIG=1 overwrites
  # config/advanced_config.json with THIS HOST's own advanced config,
  # which carries canonical_margin / reference_aspect_ratio / slice_* --
  # every one of which feeds canonical_geometry_key (crop framing) and the
  # SAHI slice-tile hash. If this host's advanced config differs from the
  # one that was PACKED, a run using this switch can produce DIFFERENT
  # tracking results than the job's own snapshot would have, AND the
  # resulting caches are keyed differently, so a cache pulled back here
  # will never hit locally against the job's original (unswitched)
  # snapshot either. Use only when you specifically intend to run this
  # job under THIS host's advanced config instead of the one it shipped
  # with.
  # ****************************************************************
  # Minor fix: $HOME/.config/hydra-suite is Linux-only (platformdirs puts
  # macOS config at "$HOME/Library/Application Support/hydra-suite"); prefer
  # HYDRA_HOST_CONFIG_DIR when set (the common case here, since run.sh always
  # sets it above) and fall back per-OS only when it is genuinely empty.
  if [ -n "${HYDRA_HOST_CONFIG_DIR:-}" ]; then
    HOST_ADV="$HYDRA_HOST_CONFIG_DIR/advanced_config.json"
  elif [ "$(uname -s)" = "Darwin" ]; then
    HOST_ADV="$HOME/Library/Application Support/hydra-suite/advanced_config.json"
  else
    HOST_ADV="$HOME/.config/hydra-suite/advanced_config.json"
  fi
  if [ -f "$HOST_ADV" ]; then
    echo "run.sh: using the HOST advanced config ($HOST_ADV), not the job snapshot"
    cp "$HOST_ADV" "$JOB/config/advanced_config.json"
    # Minor fix: `pull` never fetches config/ back (Task 9's build_push_input_list
    # is the only manifest-driven list, and pull only fetches videos/ outputs +
    # logs/), so if this branch overwrote the pushed advanced_config.json, the
    # local record silently diverges from what actually ran -- with nothing in
    # `job status`/the pulled artifacts saying so. Record it in the one place
    # that DOES travel back: the run-record line.
    export HYDRA_JOB_HOST_ADVANCED_CONFIG_USED=1
  fi
fi

# cd is what makes the job-relative videos.txt and sidecar file_path values work
# with load_video_list()'s CWD-relative semantics.
cd "$JOB"
mkdir -p logs
START="$(date -u +%FT%TZ)"

# Fix A2a: `trackerkit` must NOT be assumed to be on PATH. `ssh host 'cmd'`
# (non-interactive, non-login) does not source the shell profile that puts a
# conda env's entry points on PATH -- verified on firebrat: even
# `ssh firebrat 'bash -lc "which trackerkit"'` -> rc=1; only an explicit
# `source <conda>/etc/profile.d/conda.sh && conda activate <env>` resolves it.
# Global Constraint (CLAUDE.md line 44) forbids a bare `trackerkit` locally
# for the same reason. HYDRA_JOB_TRACKERKIT lets the caller (job_cli.py's
# `--remote-bootstrap`, or a local override) inject a fully-qualified
# invocation; the default keeps today's behavior for a shell where it IS on
# PATH (e.g. an already-activated interactive session).
#
# Minor fix (round-6): `job_cli.py` builds this value with Python's
# `shlex.quote(sys.executable)` so a space-containing interpreter path
# survives -- but `shlex.quote` produces POSIX shell-syntax quoting (wrapping
# quotes as literal characters), and a bare unquoted expansion word-splits on
# whitespace WITHOUT re-parsing embedded quote characters as syntax -- they
# stay literal -- so a quoted path with a space would word-split into two
# bogus tokens (one carrying a stray leading/trailing quote character)
# instead of being treated as one argument.
#
# Fix Y1 (round-7, REPLACES the round-6 `eval` fix below -- that fix was
# itself the bug): `eval "$TRACKERKIT track --video-list videos.txt
# $(printf '%q ' "$@")"` is broken with ZERO passthrough args. Verified by
# running it: bash's `printf '%q ' "$@"` on an EMPTY "$@" still emits one
# token, a literal `'' ` (an empty-but-quoted string), not nothing -- `"$@"`
# only expands to "nothing" when it is NOT first flattened through a command
# substitution that itself always produces at least the joining space.
# `eval` then re-parses that `''` as a real empty positional argument, so
# `trackerkit track --video-list videos.txt ''` runs, and the REAL parser
# (`parse_arguments`) rejects it: `error: use either explicit video paths or
# --video-list, not both` -> SystemExit 2. Every `job run` with no extra
# flags -- the common case, exercised by Task 13's acceptance runs -- would
# therefore die before the first frame.
#
# The actual fix does not need `eval` for the ARGV path at all. `eval` is
# only needed to turn ONE trusted, pre-quoted string (the interpreter
# invocation itself, which may be a multi-word `conda run -n hydra-mps
# trackerkit`-style string) into multiple argv words; it must never be used
# on `"$@"`, which is untrusted job-runtime argv and bash already knows how
# to pass through byte-for-byte via a real array and `"${TK[@]}"`.
TRACKERKIT_STR="${HYDRA_JOB_TRACKERKIT:-trackerkit}"
#   eval only on the trusted interpreter string -> build an array once:
eval "TK=($TRACKERKIT_STR)"
#   From here on, every invocation is "${TK[@]}" ... "$@" -- no eval, no
#   printf %q, no re-quoting. "$@" with zero elements now correctly expands
#   to NOTHING (this is the whole point of using an array + native "$@"
#   passthrough instead of round-tripping args through a string).

# Fix W1d (round-5 correction): preflight runs BEFORE `set +e`, so a failing
# preflight ABORTS under `set -euo pipefail` instead of printing and letting
# `track` proceed onto a truncated video. The previous revision placed this
# call after `set +e` while its comment claimed the opposite, which made the
# whole hand-run protection inert -- exactly the case W1d exists for.
# `job preflight .` runs from $JOB (we already cd'd) so shared aliases resolve
# from HYDRA_HOST_CONFIG_DIR as check 6 describes, and its diagnostics land in
# logs/preflight.json rather than interleaved into logs/run.log.
#
# Fix X2: this self-preflight is deliberately FLAG-LESS -- it never sees a
# one-off `--shared-root ALIAS=PATH` or `--allow-tier-fallback` the caller may
# have passed to `trackerkit job run` AS CLI FLAGS -- they arrive instead via
# HYDRA_JOB_PREFLIGHT_ARGS (see the else-branch below). `job run` (both the local and
# the ssh-chained remote branch, fix M7) already runs `preflight_job` itself,
# WITH those flags, immediately before invoking run.sh. Re-running a
# flag-less preflight here would then fail on the exact alias/tier state the
# first preflight just proved workable -- silently making --shared-root and
# --allow-tier-fallback dead for every `job run` path. So `job run` sets
# HYDRA_JOB_SKIP_PREFLIGHT=1 in run.sh's environment once ITS OWN preflight
# (with the caller's flags) has already passed; run.sh honors it here and
# skips straight to `track`. A hand-run `./run.sh` (no `job run` wrapper, the
# case W1d protects) never has this variable set, so it always gets the
# flag-less self-preflight -- the hand-run protection stays intact.
if [ "${HYDRA_JOB_SKIP_PREFLIGHT:-0}" = "1" ]; then
  echo "run.sh: skipping self-preflight (already run by 'trackerkit job run' with its flags)"
else
  # This is the ONLY preflight for a hand-run ./run.sh. It picks up overrides
  # from HYDRA_JOB_PREFLIGHT_ARGS so the hand-run path is self-sufficient for a
  # job whose videos are `shared` (spec 6.7) against an alias the box does not
  # have persisted:
  #
  #   HYDRA_JOB_PREFLIGHT_ARGS='--shared-root labnas=/mnt/lab' ./run.sh
  #
  # `trackerkit job run` sets this from its own --shared-root /
  # --allow-tier-fallback flags, so a job launched once through `job run`
  # leaves behind a run.sh a human can re-run by hand with the same result.
  # Without it a hand-run aborts with "unknown shared-root alias ...; known
  # aliases: (none configured)" even though `job run` had just succeeded --
  # and the spec calls run.sh "the executable contract", so the two paths must
  # agree. Persisting the alias with `trackerkit job shared-root add` on the
  # box is the other valid answer, and the better one for a machine that will
  # run many jobs off the same share.
  #
  # eval touches ONLY the trusted operator-supplied string (exactly as
  # TRACKERKIT_STR above), never argv, so a quoted path containing spaces
  # survives. The ${PF_ARGS[@]+...} guard keeps this correct under `set -u`
  # when the array is empty.
  eval "PF_ARGS=(${HYDRA_JOB_PREFLIGHT_ARGS:-})"
  "${TK[@]}" job preflight . ${PF_ARGS[@]+"${PF_ARGS[@]}"}
fi

set +e
# Fix Y1: "${TK[@]}" ... "$@" passes the job's own extra argv through
# NATIVELY -- no string round-trip, no printf %q, no eval. This is what
# makes a ZERO-argument "$@" expand to truly nothing (the round-6 `eval
# "... $(printf '%q ' "$@")"` form could not do this: printf on an empty
# "$@" still emits one `''` token, which eval then re-parsed as a real,
# bogus empty positional -- see the comment above TRACKERKIT_STR= for the
# full empirical trace). A video path or any other forwarded argument
# containing a space survives untouched, exactly as bash's own "$@"
# semantics guarantee, with no re-quoting step to get wrong.
"${TK[@]}" track --video-list videos.txt "$@" 2>&1 | tee -a logs/run.log
CODE=${PIPESTATUS[0]}
set -e
# Fix A2d: this whole script runs under `set -euo pipefail`. If `_record-run`
# itself fails (e.g. the same PATH issue below `$TRACKERKIT`, or a transient
# I/O error writing logs/runs.jsonl), a `set -e`-fatal exit here would replace
# $CODE -- the run's REAL exit code -- with _record-run's exit code, silently
# masking a tracking failure as a bookkeeping failure or vice versa. Make the
# record step non-fatal and always preserve and exit with the run's own $CODE.
"${TK[@]}" job _record-run --started "$START" --exit-code "$CODE" -- "$@" || \\
  echo "run.sh: WARNING: failed to append to logs/runs.jsonl (exit $?); run's own exit code $CODE is unaffected" >&2
exit "$CODE"
"""


def render_run_sh() -> str:
    return RUN_SH
