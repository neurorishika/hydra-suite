#!/usr/bin/env bash
# Put the equivalence fixtures (short clips + the models their configs
# reference) on this machine and verify them against manifest.json:
#   - clips  -> tools/equivalence/fixtures/clips/
#   - models -> this machine's hydra-suite models dir (see get_models_dir())
#
# Two sources, tried in this order:
#
#   1. PEER TRANSFER (set PEER=user@host) -- rsync both from a machine that
#      already has them. This is the supported path while the GitHub Release
#      below is unpublished.
#   2. GITHUB RELEASE -- public download URLs via curl, no `gh` CLI needed.
#
# Either way the result is verified against manifest.json: clip sha256s, and
# per-file sha256s for every model (manifest's `model_files`). Anything already
# present and matching is left alone, so re-running is cheap.
#
#   bash tools/equivalence/fixtures/fetch_fixtures.sh
#   PEER=rutalab@mehek.taild08eb9.ts.net bash tools/equivalence/fixtures/fetch_fixtures.sh
#
# Env knobs:
#   PEER             user@host of a machine that already has the fixtures
#   PEER_REPO        peer's repo checkout (default ~/hydra-suite) -- source of clips
#   PEER_MODELS_DIR  peer's models dir; auto-detected over ssh when omitted
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.json"
CLIPS_DIR="$HERE/clips"
STAGING="$HERE/staging"
mkdir -p "$CLIPS_DIR" "$STAGING"

PEER="${PEER:-}"
PEER_REPO="${PEER_REPO:-hydra-suite}"
PEER_MODELS_DIR="${PEER_MODELS_DIR:-}"

read -r REPO TAG < <(python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["repo"], m["release_tag"])
PY
)
BASE="https://github.com/$REPO/releases/download/$TAG"

MODELS_DIR=$(python - <<'PY'
from hydra_suite.paths import get_models_dir
print(get_models_dir())
PY
)
if [ -z "$MODELS_DIR" ]; then
  echo "!! could not resolve this machine's models dir (is hydra_suite importable?)" >&2
  exit 1
fi
mkdir -p "$MODELS_DIR"
echo "### fixtures: clips -> $CLIPS_DIR"
echo "###           models -> $MODELS_DIR"

verify() {  # file expected_sha
  local got
  got=$(shasum -a 256 "$1" 2>/dev/null | awk '{print $1}')
  [ "$got" = "$2" ]
}

fetch() {  # url dest sha
  if [ -f "$2" ] && verify "$2" "$3"; then
    echo "  ok (cached): $(basename "$2")"; return 0
  fi
  echo "  downloading $(basename "$2") ..."
  curl -fL --retry 3 -o "$2" "$1" || { echo "!! download failed: $1" >&2; return 1; }
  if ! verify "$2" "$3"; then
    echo "!! checksum mismatch: $2" >&2; return 1
  fi
}

# ---------------------------------------------------------------- manifest IO
clip_rows() {  # name|sha256
  python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for c in m["clips"]:
    print(f"{c['name']}|{c['sha256']}")
PY
}

model_rows() {  # relpath|sha256
  python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for e in m.get("model_files", []):
    print(f"{e['path']}|{e['sha256']}")
PY
}

# Which models are missing or corrupt right now. Printed one relative path per
# line so it can feed rsync --files-from directly.
missing_models() {
  while IFS='|' read -r rel sha; do
    [ -n "$rel" ] || continue
    if ! verify "$MODELS_DIR/$rel" "$sha"; then
      printf '%s\n' "$rel"
    fi
  done < <(model_rows)
}

# --------------------------------------------------------------- peer transfer
if [ -n "$PEER" ]; then
  echo "### peer transfer from $PEER"
  if [ -z "$PEER_MODELS_DIR" ]; then
    # The peer's models dir is platform-specific, so ask it. This needs a
    # hydra env active in the peer's login shell; if it is not, the user can
    # pass PEER_MODELS_DIR explicitly.
    PEER_MODELS_DIR=$(ssh "$PEER" 'python -c "from hydra_suite.paths import get_models_dir; print(get_models_dir())"' 2>/dev/null | tail -1)
  fi
  if [ -z "$PEER_MODELS_DIR" ]; then
    echo "!! could not resolve the peer's models dir over ssh." >&2
    echo "   Re-run with it named explicitly, e.g." >&2
    echo "   PEER=$PEER PEER_MODELS_DIR='~/.local/share/hydra-suite/models' bash \$0" >&2
    exit 1
  fi
  echo "  peer models dir: $PEER_MODELS_DIR"

  # Clips are gitignored, so they come from the peer's checkout, not from git.
  rsync -a --info=stats1 "$PEER:$PEER_REPO/tools/equivalence/fixtures/clips/" "$CLIPS_DIR/" \
    || { echo "!! rsync of clips from $PEER failed" >&2; exit 1; }

  # Models: copy only what is missing or corrupt here, by exact manifest path.
  NEED_LIST="$STAGING/.models_needed"
  missing_models > "$NEED_LIST"
  if [ -s "$NEED_LIST" ]; then
    echo "  fetching $(wc -l < "$NEED_LIST" | tr -d ' ') model file(s) from peer"
    rsync -a --files-from="$NEED_LIST" "$PEER:$PEER_MODELS_DIR/" "$MODELS_DIR/" \
      || { echo "!! rsync of models from $PEER failed" >&2; rm -f "$NEED_LIST"; exit 1; }
  else
    echo "  models already complete; nothing to copy"
  fi
  rm -f "$NEED_LIST"
fi

# ----------------------------------------------------------------------- clips
# NOTE: the loop body runs in a subshell (RHS of a pipe), so an `exit 1` in
# there only kills the subshell, not this script — a 404 would otherwise be
# swallowed and the script would sail on to "fixtures ready", leaving pose
# clips to silently produce empty CSVs that then falsely compare as
# EQUIVALENT. Route failures through a status file instead.
CLIP_FAIL_MARKER="$STAGING/.clip_fetch_failed"
rm -f "$CLIP_FAIL_MARKER"
clip_rows | while IFS='|' read -r name sha; do
  fetch "$BASE/$name" "$CLIPS_DIR/$name" "$sha" || { touch "$CLIP_FAIL_MARKER"; }
done
if [ -f "$CLIP_FAIL_MARKER" ]; then
  rm -f "$CLIP_FAIL_MARKER"
  cat >&2 <<EOF

!! fixture clips are missing and could not be downloaded. As of 2026-09 the
   release this script points at ($TAG in $REPO) 404s -- it needs
   republishing; do not assume a passing run here means fixtures exist.

   Use the peer-transfer path instead, from a machine that already has them:

     PEER=user@host bash tools/equivalence/fixtures/fetch_fixtures.sh

   (add PEER_REPO=/path/to/hydra-suite if the peer's checkout is not
   ~/hydra-suite, and PEER_MODELS_DIR=... if this script cannot resolve the
   peer's models dir over ssh).

   ABORTING rather than continuing with partial/missing fixtures.
EOF
  exit 1
fi

# ---------------------------------------------------------------------- models
# Verify the models dir directly against the manifest's per-file checksums. The
# archive is only a transport: when every model is already present and correct
# there is nothing to download, which is what makes a peer-transferred or
# hand-copied models dir a first-class, checkable source. Falling back to the
# archive on any miss also means a stale/unreachable archive can no longer
# block a machine whose models are fine.
MISSING="$(missing_models)"
if [ -z "$MISSING" ]; then
  echo "### models verified against manifest (all $(model_rows | wc -l | tr -d ' ') files match)"
else
  echo "### $(printf '%s\n' "$MISSING" | wc -l | tr -d ' ') model file(s) missing or corrupt; fetching archive"
  read -r MNAME MSHA < <(python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["models_archive"]["name"], m["models_archive"]["sha256"])
PY
)
  if ! fetch "$BASE/$MNAME" "$STAGING/$MNAME" "$MSHA"; then
    cat >&2 <<EOF

!! these model files are missing or corrupt under $MODELS_DIR:
$(printf '   - %s\n' $MISSING)

   The release archive could not be downloaded or did not match its manifest
   checksum, so it cannot repair them. Copy the models from a machine that has
   them:

     PEER=user@host bash tools/equivalence/fixtures/fetch_fixtures.sh

   ABORTING rather than continuing with partial/missing models: a missing
   model does not fail loudly, it produces empty CSVs that then compare as
   EQUIVALENT against each other.
EOF
    exit 1
  fi
  echo "### extracting $MNAME -> $MODELS_DIR"
  tar -xzf "$STAGING/$MNAME" -C "$MODELS_DIR"

  STILL_MISSING="$(missing_models)"
  if [ -n "$STILL_MISSING" ]; then
    echo "!! the archive did not supply every model the manifest lists:" >&2
    printf '   - %s\n' $STILL_MISSING >&2
    echo "   The archive is stale relative to manifest.json; re-run make_manifest.py" >&2
    echo "   on a machine with a complete models dir, or use PEER= transfer." >&2
    exit 1
  fi
  echo "### models verified against manifest"
fi

echo "### fixtures ready. Run:  bash tools/equivalence/run_matrix.sh   (FIXTURES=1 is the default target set)"
