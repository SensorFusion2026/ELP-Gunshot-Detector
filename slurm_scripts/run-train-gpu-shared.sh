#!/usr/bin/env bash
#SBATCH --job-name=elp_gunshot_train
#SBATCH --account=cso100
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/expanse/lustre/projects/cso100/%u/ElephantListeningProject/ELP-Gunshot-Detector/slurm_logs/%x.o%j.%N

PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/cso100/$USER/ElephantListeningProject}"
REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/ELP-Gunshot-Detector}"

exec bash "$REPO_ROOT/slurm_scripts/_run-train-gpu.sh" "$@"