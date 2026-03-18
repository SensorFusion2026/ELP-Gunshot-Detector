#!/usr/bin/env bash

set -euo pipefail

module purge
module load singularitypro/3.11
module list

export PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/cso100/$USER/ElephantListeningProject}"
export REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/ELP-Gunshot-Detector}"
export SIF="${SIF:-$PROJECT_ROOT/tensorflow-2.15.0-gpu.sif}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$PROJECT_ROOT/.pythonuserbase}"
export NVIDIA_DISABLE_REQUIRE=true

STAMP_DIR="$PYTHONUSERBASE/.elp_gunshot"
STAMP_FILE="$STAMP_DIR/pyproject.sha256"

mkdir -p "$STAMP_DIR"

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: Repo not found: $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: Container not found: $SIF" >&2
  exit 1
fi

CURRENT_HASH="$(sha256sum "$REPO_ROOT/pyproject.toml" | awk '{print $1}')"

if [[ -f "$STAMP_FILE" && "$(cat "$STAMP_FILE")" == "$CURRENT_HASH" ]]; then
  echo "Setup already up to date (pyproject.toml hash matches). Skipping reinstall."
  exit 0
fi

singularity exec --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  python -m pip install --upgrade --user pip setuptools wheel && \
  python -m pip install --user -e '$REPO_ROOT'
"

printf '%s\n' "$CURRENT_HASH" > "$STAMP_FILE"

echo "Setup complete."
echo "Recorded pyproject.toml hash: $CURRENT_HASH"
