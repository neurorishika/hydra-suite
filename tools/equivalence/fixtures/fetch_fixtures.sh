#!/usr/bin/env bash
# Download the equivalence fixtures (short clips + required models) from the
# GitHub Release named in manifest.json, verify checksums, and place them:
#   - clips      -> tools/equivalence/fixtures/clips/
#   - models.tar -> extracted into this machine's hydra-suite models dir
#
# Works without the `gh` CLI (public-release download URLs via curl). Re-running
# is cheap: assets with a matching sha256 are skipped.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/manifest.json"
CLIPS_DIR="$HERE/clips"
STAGING="$HERE/staging"
mkdir -p "$CLIPS_DIR" "$STAGING"

read -r REPO TAG < <(python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["repo"], m["release_tag"])
PY
)
BASE="https://github.com/$REPO/releases/download/$TAG"
echo "### fetching fixtures from $BASE"

verify() {  # file expected_sha
  local got
  got=$(shasum -a 256 "$1" | awk '{print $1}')
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

# Clips
python - "$MANIFEST" <<'PY' | while IFS='|' read -r name sha; do
import json, sys
m = json.load(open(sys.argv[1]))
for c in m["clips"]:
    print(f"{c['name']}|{c['sha256']}")
PY
  fetch "$BASE/$name" "$CLIPS_DIR/$name" "$sha" || exit 1
done

# Models archive -> extract into models dir
read -r MNAME MSHA < <(python - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(m["models_archive"]["name"], m["models_archive"]["sha256"])
PY
)
fetch "$BASE/$MNAME" "$STAGING/$MNAME" "$MSHA" || exit 1

MODELS_DIR=$(python - <<'PY'
from hydra_suite.paths import get_models_dir
print(get_models_dir())
PY
)
echo "### extracting $MNAME -> $MODELS_DIR"
mkdir -p "$MODELS_DIR"
tar -xzf "$STAGING/$MNAME" -C "$MODELS_DIR"
echo "### fixtures ready. Run:  bash tools/equivalence/run_matrix.sh   (FIXTURES=1 is the default target set)"
