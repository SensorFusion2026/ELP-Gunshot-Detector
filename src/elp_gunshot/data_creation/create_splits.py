# src/elp_gunshot/data_creation/create_splits.py
# Usage: python -m elp_gunshot.data_creation.create_splits

from pathlib import Path
import pandas as pd
import numpy as np
from elp_gunshot.config.paths import CLIPS_PLAN_CSV, SPLITS_DIR

# -----------------------
# Paths (centralized in config.paths)
# -----------------------
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Settings
# -----------------------
SEED = 0
RNG = np.random.default_rng(SEED)

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
# test = remainder

MODEL1_CAPS = {
    "korup":   {"pos": 60, "neg": 60},
    "ecoguns": {"pos": 60, "neg": 60},
}

MODEL2_FRAC = 0.5
MODEL3_FRAC = 1.0

# -----------------------
# Load clip plan
# -----------------------
if not CLIPS_PLAN_CSV.exists():
    raise FileNotFoundError(f"Missing {CLIPS_PLAN_CSV}. Run create_clips_plan.py first.")

plan = pd.read_csv(CLIPS_PLAN_CSV)

required = {"label", "location", "source_wav_relpath", "clip_wav_relpath"}
missing = required - set(plan.columns)
if missing:
    raise ValueError(f"clips_plan.csv missing columns: {sorted(missing)}")

# -----------------------
# Helper functions
# -----------------------
def subsample_by_wav(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """
    Subsample the dataset by selecting a fraction of unique source WAV files
    and keeping all clips that originate from those WAVs.

    This is used to control dataset size for different model versions (e.g.,
    Model 2 uses 50% of available WAVs), while avoiding clip-level leakage.
    """
    wavs = df["source_wav_relpath"].drop_duplicates().to_numpy()
    RNG.shuffle(wavs)

    k = max(1, int(frac * len(wavs)))
    keep = set(wavs[:k])
    return df[df["source_wav_relpath"].isin(keep)].copy()

def split_by_wav(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign train/validation/test splits at the source WAV level.

    All clips from the same source WAV are placed in the same split to prevent
    background, burst, and recording-condition leakage across splits.
    """
    wavs = df["source_wav_relpath"].drop_duplicates().to_numpy()
    RNG.shuffle(wavs)

    n = len(wavs)
    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)

    train_wavs = set(wavs[:n_train])
    val_wavs = set(wavs[n_train:n_train + n_val])

    out = df.copy()
    out["split"] = "test"
    out.loc[out["source_wav_relpath"].isin(train_wavs), "split"] = "train"
    out.loc[out["source_wav_relpath"].isin(val_wavs), "split"] = "val"
    return out

def build_model1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a small feasibility dataset using absolute caps.
    """
    parts = []
    for loc, caps in MODEL1_CAPS.items():
        for label, cap in caps.items():
            sub = df[(df["location"] == loc) & (df["label"] == label)]
            if sub.empty:
                continue
            sub = sub.sample(frac=1.0, random_state=SEED)
            parts.append(sub.head(cap))

    if not parts:
        return df.head(0).copy()

    return pd.concat(parts, ignore_index=True)


def write_split(df: pd.DataFrame, path: Path):
    cols = ["split", "label", "location", "source_wav_relpath", "clip_wav_relpath"]
    df = df[cols].sort_values(cols)
    df.to_csv(path, index=False)
    print(f"Wrote {path} ({len(df)} rows)")


# -----------------------
# Build splits
# -----------------------

# Model 1: feasibility (small amount of data)
model1_base = build_model1(plan)
model1 = split_by_wav(model1_base)

# Model 2: scalability (50% of data)
model2_base = subsample_by_wav(plan, MODEL2_FRAC)
model2 = split_by_wav(model2_base)

# Model 3: performance (full data)
model3_base = subsample_by_wav(plan, MODEL3_FRAC)
model3 = split_by_wav(model3_base)

# -----------------------
# Write outputs
# -----------------------
write_split(model1, SPLITS_DIR / "model1.csv")
write_split(model2, SPLITS_DIR / "model2.csv")
write_split(model3, SPLITS_DIR / "model3.csv")

# -----------------------
# Print summaries
# -----------------------
def summarize(df, name):
    print(f"\n{name}")
    print(df.groupby(["split", "location", "label"]).size())

summarize(model1, "MODEL 1 (Feasibility)")
summarize(model2, "MODEL 2 (Scaled)")
summarize(model3, "MODEL 3 (Performance)")