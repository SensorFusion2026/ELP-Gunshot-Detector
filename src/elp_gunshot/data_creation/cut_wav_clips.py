# src/elp_gunshot/data_creation/cut_wav_clips.py
# Usage: python -m elp_gunshot.data_creation.cut_wav_clips

import wave

import numpy as np
import pandas as pd

from elp_gunshot.config.paths import CLIPS_PLAN_CSV, RAW_ROOT, WAV_CLIPS_ROOT

# PLAN_CSV = CLIPS_PLAN_CSV
OUT_ROOT = WAV_CLIPS_ROOT  # data/wav_clips

if not CLIPS_PLAN_CSV.exists():
    raise FileNotFoundError(f"Missing plan file: {CLIPS_PLAN_CSV}. Run create_clips_plan.py first.")

plan = pd.read_csv(CLIPS_PLAN_CSV)

required = {"label", "location", "source_wav_relpath", "start_s", "duration_s", "clip_wav_relpath"}
missing = required - set(plan.columns)
if missing:
    raise ValueError(f"clips_plan.csv missing columns: {sorted(missing)}")

saved = 0
skipped_exists = 0
skipped_missing_wav = 0
skipped_bad_format = 0
skipped_short = 0

for i, row in plan.iterrows():
    source_rel = str(row["source_wav_relpath"])
    start_s = float(row["start_s"])
    dur_s = float(row["duration_s"])

    wav_path = RAW_ROOT / source_rel
    out_path = OUT_ROOT / str(row["clip_wav_relpath"])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        skipped_exists += 1
        continue

    if not wav_path.exists():
        skipped_missing_wav += 1
        continue

    with wave.open(str(wav_path), "rb") as w:
        # Keep strict assumptions (same as clip planning)
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            skipped_bad_format += 1
            continue

        sr = w.getframerate()
        start_frame = int(sr * start_s)
        n_frames = int(sr * dur_s)

        # Set start position and error check: fail early if the 
        # requested start position is past EOF.
        try:
            w.setpos(start_frame)
        except wave.Error:
            skipped_short += 1
            continue

        frames = w.readframes(n_frames)

    audio = np.frombuffer(frames, dtype=np.int16)

    # Ensure full clip length
    if len(audio) != n_frames:
        skipped_short += 1
        continue

    # Write output wav
    with wave.open(str(out_path), "wb") as out_w:
        out_w.setnchannels(1)
        out_w.setsampwidth(2)
        out_w.setframerate(sr)
        out_w.writeframes(audio.tobytes())

    saved += 1

print("\n=== Done cutting clips ===")
print(f"Saved:                      {saved}")
print(f"Skipped (exists):           {skipped_exists}")
print(f"Skipped (missing wav):      {skipped_missing_wav}")
print(f"Skipped (bad mono/int16):   {skipped_bad_format}")
print(f"Skipped (short/EOF):        {skipped_short}")
print(f"Output root:                {OUT_ROOT}")