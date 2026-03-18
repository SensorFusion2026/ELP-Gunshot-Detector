# config/paths.py
"""
paths.py

Centralized path configuration for the ELP Gunshot Detector.

Current path policy:
- Versioned manifests live in `src/elp_gunshot/data_creation/`.
- Generated artifacts (wav clips and TFRecords) live under `data/`.
- Raw Cornell source paths are resolved only when `ENVIRONMENT=local`.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------- Project root (repo root) -----------
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Get environment variables from .env located at the project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

ENVIRONMENT = os.getenv("ENVIRONMENT", "remote")
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")

# ---------------------------------------------------------------------
# Raw Cornell data roots (local only)
# ---------------------------------------------------------------------

if ENVIRONMENT == "local":
    if not CORNELL_DATA_ROOT:
        raise ValueError(
            "CORNELL_DATA_ROOT not found. Create a .env in the project root and add:\n"
            'CORNELL_DATA_ROOT="/path/to/your/data"\n'
            'ENVIRONMENT="local"'
        )

    RAW_ROOT = Path(CORNELL_DATA_ROOT)

    PNNN_ROOT = RAW_ROOT / "Gunshot" / "Training" / "pnnn_dep1-7"
    KORUP_ROOT = RAW_ROOT / "Gunshot" / "Testing" / "Korup"
    ECOGUNS_ROOT = RAW_ROOT / "Gunshot" / "Training" / "ecoguns"

    PNNN_METADATA = PNNN_ROOT / "nn_Grid50_guns_dep1-7_Guns_Training.txt"
    PNNN_SOUNDS = PNNN_ROOT / "Sounds"

    KORUP_METADATA = KORUP_ROOT / "Korup_4kHz_Gunshots_Merged.txt"
    KORUP_SOUNDS = KORUP_ROOT / "Sounds"

    ECOGUNS_METADATA = ECOGUNS_ROOT / "Guns_Training_ecoGuns_SST.txt"
    ECOGUNS_SOUNDS = ECOGUNS_ROOT / "Sounds"

else:
    RAW_ROOT = None

    PNNN_ROOT = None
    KORUP_ROOT = None
    ECOGUNS_ROOT = None

    PNNN_METADATA = None
    PNNN_SOUNDS = None

    KORUP_METADATA = None
    KORUP_SOUNDS = None

    ECOGUNS_METADATA = None
    ECOGUNS_SOUNDS = None

# ---------------------------------------------------------------------
# Repository-managed data directories
# ---------------------------------------------------------------------

# Versioned manifests for reproducible data creation
DATA_CREATION_ROOT = PROJECT_ROOT / "src" / "elp_gunshot" / "data_creation"
CLIPS_PLAN_CSV = DATA_CREATION_ROOT / "clips_plan.csv"
SPLITS_DIR = DATA_CREATION_ROOT / "splits"

# Generated artifacts (not versioned)
DATA_ROOT = PROJECT_ROOT / "data"
WAV_CLIPS_ROOT = DATA_ROOT / "wav_clips"
TFRECORDS_ROOT = DATA_ROOT / "tfrecords"

# Training run outputs
RUNS_DIR = PROJECT_ROOT / "runs"


# ---------------------------------------------------------------------
# Utility: create derived directories if needed
# ---------------------------------------------------------------------

def ensure_directories() -> None:
    """
    Create required repo-managed directories if they do not exist.
    Safe to call at the beginning of preprocessing/training scripts.
    """
    for p in [
        DATA_ROOT,
        WAV_CLIPS_ROOT,
        SPLITS_DIR,
        TFRECORDS_ROOT,
        RUNS_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)