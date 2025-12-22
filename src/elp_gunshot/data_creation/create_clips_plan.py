# src/elp_gunshot/data_creation/create_clips_plan.py
# Usage: python -m elp_gunshot.data_creation.create_clips_plan

import numpy as np
import pandas as pd
import wave
from pathlib import Path

from elp_gunshot.config.paths import *

# -----------------------
# Settings
# -----------------------
CLIP_LEN_S = 4.0
POS_PRE_PADDING_S = 0.1
BURST_COOLDOWN_S = CLIP_LEN_S

BUFFER_S = 10.0         # negatives must be >=10s away from any positive window
NEG_PER_POS = 3         # per saved positive, per WAV
RNG = np.random.default_rng(0)

# Save clips plan next to the creation scripts (committed file)
PLAN_CSV = Path(__file__).resolve().parent / "clips_plan.csv"

# -----------------------
# Safety check: clip plan overwrite
# -----------------------
if PLAN_CSV.exists():
    raise RuntimeError(
        f"\n⚠️  clip_plan.csv already exists at:\n"
        f"    {PLAN_CSV}\n\n"
        f"This file is the source of truth for clip generation and is shared\n"
        f"across the team. Re-creating it will require everyone to re-cut\n"
        f"WAV clips and rebuild downstream artifacts (splits, TFRecords).\n\n"
        f"If you REALLY intend to regenerate the clip plan:\n"
        f"  1) Delete this file manually\n"
        f"  2) Inform the rest of the team\n"
        f"  3) Re-run create_clips_plan.py\n"
        f"  4) Re-generate WAV clips, split files, and TFRecords\n"
    )

datasets = [
    ("korup", KORUP_METADATA, KORUP_SOUNDS),
    ("ecoguns", ECOGUNS_METADATA, ECOGUNS_SOUNDS),
    ("pnnn", PNNN_METADATA, PNNN_SOUNDS)
]

rows = []

for location, metadata_path, sounds_dir in datasets:
    print(f"\n=== Planning {location.upper()} ===")

    df = pd.read_csv(metadata_path, delimiter="\t")
    if "Begin File" not in df.columns:
        raise ValueError(f"{location}: Missing 'Begin File' column in {metadata_path}")
    if "File Offset (s)" not in df.columns:
        raise ValueError(f"{location}: Missing 'File Offset (s)' column in {metadata_path}")

    df = df.sort_values(["Begin File", "File Offset (s)"])

    # For burst handling + negative exclusion
    last_pos_start_by_wav = {}
    pos_windows_by_wav = {}      # source_wav_relpath -> list of (pos_start, pos_end)
    pos_saved_count_by_wav = {}  # source_wav_relpath -> count of planned positives

    # -----------------------
    # POS plan
    # -----------------------
    for i, row in df.iterrows():
        wav_name = str(row["Begin File"])
        wav_path = Path(sounds_dir) / wav_name

        if not wav_path.exists():
            continue

        clip_start = max(0.0, float(row["File Offset (s)"]) - POS_PRE_PADDING_S)

        # Get wav path relative to the raw data root for portability
        source_wav_relpath = str(wav_path.resolve().relative_to(RAW_ROOT.resolve()))

        prev = last_pos_start_by_wav.get(source_wav_relpath)
        if prev is not None and (clip_start - prev) < BURST_COOLDOWN_S:
            continue
        last_pos_start_by_wav[source_wav_relpath] = clip_start

        # Record windows, used later for exclusions when planning neg clips
        pos_windows_by_wav.setdefault(source_wav_relpath, []).append((clip_start, clip_start + CLIP_LEN_S))
        pos_saved_count_by_wav[source_wav_relpath] = pos_saved_count_by_wav.get(source_wav_relpath, 0) + 1

        out_wav_relpath = f"pos/{location}/{wav_path.stem}_pos_{int(round(clip_start))}_{i}.wav"

        rows.append({
            "label": "pos",
            "location": location,
            "source_wav_relpath": source_wav_relpath,
            "start_s": float(clip_start),
            "duration_s": float(CLIP_LEN_S),
            "clip_wav_relpath": out_wav_relpath,
        })

    print(f"Planned {sum(pos_saved_count_by_wav.values())} pos clips for {location}")

    # -----------------------
    # NEG plan
    # -----------------------
    for source_wav_relpath, windows in pos_windows_by_wav.items():
        wav_abs = RAW_ROOT / source_wav_relpath
        if not wav_abs.exists():
            continue

        n_pos = pos_saved_count_by_wav.get(source_wav_relpath, 0)
        if n_pos == 0:
            continue

        target_negs = NEG_PER_POS * n_pos

        with wave.open(str(wav_abs), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                continue
            sr = w.getframerate()
            total_s = w.getnframes() / sr

        # forbidden zones around pos windows
        forbidden = []
        for start, end in windows:
            forbidden.append((max(0.0, start - BUFFER_S), min(total_s, end + BUFFER_S)))

        # merge forbidden
        forbidden.sort()
        merged = []
        for start, end in forbidden:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        merged = [(a, b) for a, b in merged]

        # allowed intervals
        allowed = []
        t = 0.0
        for start, end in merged:
            if start - t >= CLIP_LEN_S:
                allowed.append((t, start))
            t = max(t, end)
        if total_s - t >= CLIP_LEN_S:
            allowed.append((t, total_s))
        if not allowed:
            continue

        # discretize allowed into unique start times
        starts = []
        for a, b in allowed:
            end = b - CLIP_LEN_S
            start = a
            while start <= end:
                starts.append(start)
                start += CLIP_LEN_S
        if not starts:
            continue

        starts = np.array(starts, dtype=np.float64)
        k = min(target_negs, starts.size)
        chosen = RNG.choice(starts, size=k, replace=False)

        wav_stem = Path(source_wav_relpath).stem
        for j, neg_start in enumerate(chosen, start=1):
            out_relpath = f"neg/{location}/{wav_stem}_neg_{int(neg_start*1000)}_{j}.wav"
            rows.append({
                "label": "neg",
                "location": location,
                "source_wav_relpath": source_wav_relpath,
                "start_s": float(neg_start),
                "duration_s": float(CLIP_LEN_S),
                "clip_wav_relpath": out_relpath,
            })

    # rough neg count for this location
    neg_count = sum(1 for r in rows if r["label"] == "neg" and r["location"] == location)
    print(f"Planned {neg_count} neg clips for {location}")

# -----------------------
# Write plan CSV
# -----------------------
plan_df = pd.DataFrame(rows)

PLAN_CSV.parent.mkdir(parents=True, exist_ok=True)
plan_df.to_csv(PLAN_CSV, index=False)

print(f"\nWrote plan: {PLAN_CSV}")
print(f"Total planned clips: {len(plan_df)}")