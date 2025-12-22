# config/paths.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT")
CORNELL_DATA_ROOT = os.getenv("CORNELL_DATA_ROOT")

# Add a check to make sure the path was found
if ENVIRONMENT == "local" and not CORNELL_DATA_ROOT:
    print("❌ Error: CORNELL_DATA_ROOT not found.")
    print("Please create a .env file in the project root and add the line:")
    print('CORNELL_DATA_ROOT="/path/to/your/data"')
    exit() # Stop the script if the path isn't configured

# ---------- Project root -----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------- Raw data roots ----------
RAW_ROOT = Path(CORNELL_DATA_ROOT)

PNNN_ROOT = RAW_ROOT / "Gunshot/Training/pnnn_dep1-7"
KORUP_ROOT = RAW_ROOT / "Gunshot/Testing/Korup"
ECOGUNS_ROOT = RAW_ROOT / "Gunshot/Training/ecoguns"

PNNN_METADATA = PNNN_ROOT / "nn_Grid50_guns_dep1-7_Guns_Training.txt"
PNNN_SOUNDS = PNNN_ROOT / "Sounds"

KORUP_METADATA = KORUP_ROOT / "Korup_4kHz_Gunshots_Merged.txt"
KORUP_SOUNDS = KORUP_ROOT / "Sounds"

ECOGUNS_METADATA = ECOGUNS_ROOT / "Guns_Training_ecoGuns_SST.txt"
ECOGUNS_SOUNDS = ECOGUNS_ROOT / "Sounds"

# ---------- Derived data ----------
DATA_ROOT = PROJECT_ROOT / "data"

WAV_CLIPS_ROOT = DATA_ROOT / "wav_clips"
SPLITS_ROOT = DATA_ROOT / "splits"
TFRECORDS_ROOT = DATA_ROOT / "tfrecords"