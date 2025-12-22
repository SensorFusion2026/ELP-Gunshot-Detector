# data_creation/create_wav_clips.py
# usage (from project repo root): python -m elp_gunshot.data_creation.create_wav_clips

from pathlib import Path
import pandas as pd
import numpy as np
import wave

from elp_gunshot.config.paths import *

CLIP_LEN_S = 4.0 # seconds

datasets = [
    ("pnnn", PNNN_METADATA, PNNN_SOUNDS),
    ("korup", KORUP_METADATA, KORUP_SOUNDS),
    ("ecoguns", ECOGUNS_METADATA, ECOGUNS_SOUNDS)
]

for name, metadata_path, sounds_dir in datasets:
    print(f"\n=== Processing {name.upper()} ===")
    successful_clips = 0

    out_dir = WAV_CLIPS_ROOT / "pos" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_path, delimiter="\t")

    if "Begin File" not in df.columns:
        raise ValueError("Missing 'Begin File' column in metadata")
    if "File Offset (s)" not in df.columns:
        raise ValueError("Missing 'File Offset (s)' column in metadata")

    df = df.sort_values(["Begin File", "File Offset (s)"])

    last_start_by_wav = {}

    for idx, row in df.iterrows():
        input_wav_name = row["Begin File"]
        input_wav_path = sounds_dir / input_wav_name

        if not input_wav_path.exists():
            # print(f"Missing WAV: {input_wav_path}")
            continue

        clip_start = max(0.0, float(row["File Offset (s)"]) - 0.1)

        prev = last_start_by_wav.get(str(input_wav_path), None)
        if prev is not None and (clip_start - prev) < CLIP_LEN_S:
            continue
        last_start_by_wav[str(input_wav_path)] = clip_start

        out_name = f"{input_wav_path.stem}_pos_{int(round(clip_start))}_{idx}.wav"
        out_path = out_dir / out_name

        with wave.open(str(input_wav_path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                print("Skipping non-mono or non-int16:", input_wav_path)
                continue
            sr = w.getframerate()
            w.setpos(int(sr * clip_start))
            frames = w.readframes(int(sr * CLIP_LEN_S))

        # Safety check for exactly CLIP_LEN_S seconds worth of int16 samples
        audio = np.frombuffer(frames, dtype=np.int16)
        if len(audio) != int(sr * CLIP_LEN_S):
            print(f"Skipping short clip: {out_name}")
            continue

        with wave.open(str(out_path), "wb") as out_w:
            out_w.setnchannels(1)
            out_w.setsampwidth(2)
            out_w.setframerate(sr)
            out_w.writeframes(audio.tobytes())
        
        successful_clips += 1

    print(f"{successful_clips} clips saved to {out_dir}")