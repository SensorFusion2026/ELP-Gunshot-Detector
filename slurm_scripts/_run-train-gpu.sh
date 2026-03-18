#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<EOF >&2
Usage: $0 <model1|model2|model3> [epochs]
  model1..model3  picks a dataset split definition
  epochs          optional positive integer epoch count override
EOF
  exit 1
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  usage
fi

MODEL="${1,,}"
EPOCHS="${2:-}"

if [[ "$MODEL" != "model1" && "$MODEL" != "model2" && "$MODEL" != "model3" ]]; then
  echo "MODEL must be one of model1, model2, model3" >&2
  usage
fi

if [[ -n "$EPOCHS" && ! "$EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "EPOCHS must be a positive integer" >&2
  usage
fi

module purge
module load singularitypro/3.11
module list

export PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/cso100/$USER/ElephantListeningProject}"
export REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/ELP-Gunshot-Detector}"
export SIF="${SIF:-$PROJECT_ROOT/tensorflow-2.15.0-gpu.sif}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-$PROJECT_ROOT/.pythonuserbase}"
export NVIDIA_DISABLE_REQUIRE=true

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "ERROR: Repo not found: $REPO_ROOT" >&2
  exit 1
fi

SLURM_LOG_DIR="$REPO_ROOT/slurm_logs"
if [[ ! -d "$SLURM_LOG_DIR" ]]; then
  echo "ERROR: $SLURM_LOG_DIR does not exist." >&2
  echo "Create it before submitting the job:" >&2
  echo "  mkdir -p $SLURM_LOG_DIR" >&2
  exit 1
fi

if [[ ! -f "$SIF" ]]; then
  echo "ERROR: Container not found: $SIF" >&2
  exit 1
fi

STAMP_DIR="$PYTHONUSERBASE/.elp_gunshot"
STAMP_FILE="$STAMP_DIR/pyproject.sha256"

CURRENT_HASH="$(sha256sum "$REPO_ROOT/pyproject.toml" | awk '{print $1}')"
INSTALLED_HASH=""

if [[ -f "$STAMP_FILE" ]]; then
  INSTALLED_HASH="$(cat "$STAMP_FILE")"
fi

if [[ ! -f "$STAMP_FILE" || "$CURRENT_HASH" != "$INSTALLED_HASH" ]]; then
  echo "ERROR: Python user environment is missing or out of date." >&2
  echo "Run setup first:" >&2
  echo "  $REPO_ROOT/slurm_scripts/setup-pythonuserbase.sh" >&2
  exit 1
fi

GPU_CHECK_SCRIPT="python - <<'PY'
import sys
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    sys.exit('No GPUs detected inside container')
print('Detected GPUs:', gpus)
PY"

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "$GPU_CHECK_SCRIPT"

EPOCH_EXPORT=""
[ -n "$EPOCHS" ] && EPOCH_EXPORT="export EPOCHS='$EPOCHS' && \
"

singularity exec --nv --bind "$PROJECT_ROOT:$PROJECT_ROOT:rw" "$SIF" bash -lc "\
  export PYTHONUSERBASE='$PYTHONUSERBASE' && \
  export PATH='$PYTHONUSERBASE/bin:'\"\$PATH\" && \
  export MODEL='$MODEL' && \
  ${EPOCH_EXPORT} \
  python -m elp_gunshot.train_cnn \
"