#!/usr/bin/env bash
#SBATCH --job-name=elp_gunshot_train_debug
#SBATCH --account=cso100
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --output=/expanse/lustre/projects/cso100/%u/ElephantListeningProject/ELP-Gunshot-Detector/slurm_logs/%x.o%j.%N

PROJECT_ROOT="${PROJECT_ROOT:-/expanse/lustre/projects/cso100/$USER/ElephantListeningProject}"
REPO_ROOT="${REPO_ROOT:-$PROJECT_ROOT/ELP-Gunshot-Detector}"

exec bash "$REPO_ROOT/slurm_scripts/_run-train-gpu.sh" "$@"