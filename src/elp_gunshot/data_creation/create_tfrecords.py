# src/elp_gunshot/data_creation/create_tfrecords.py
# Usage:
#   python -m elp_gunshot.data_creation.create_tfrecords
#
# Optional environment variables:
#   MODEL=model1|model2|model3      (default: model1)
#   MASK=nomask | bp<low>_<high>    (default: nomask)
#       optional frequency mask (bandpass filter); 0 <= low <= high <= 2000 Hz
#
# Examples:
#   MODEL=model2 python -m elp_gunshot.data_creation.create_tfrecords
#   MODEL=model3 MASK=bp100_1800 python -m elp_gunshot.data_creation.create_tfrecords
#   MASK=bp150_1600 python -m elp_gunshot.data_creation.create_tfrecords

import pandas as pd
import tensorflow as tf
import os
import re

from elp_gunshot.config.paths import SPLITS_DIR, TFRECORDS_ROOT, WAV_CLIPS_ROOT

# -----------------------
# CONFIG
# -----------------------
MODEL = os.getenv("MODEL", "model1").strip()
if MODEL not in ("model1", "model2", "model3"):
    raise ValueError('Invalid MODEL. Use MODEL=model1|model2|model3.')

MASK = os.getenv("MASK", "nomask").strip()

CLIP_LEN_S = 4.0
TARGET_SR = 4000  # standardize to 4kHz
EXPECTED_SAMPLES = int(TARGET_SR * CLIP_LEN_S)
NYQUIST_HZ = TARGET_SR // 2  # 2000 for 4kHz

# STFT params (tune later)
FRAME_LENGTH = 256
FRAME_STEP = 128
FFT_LENGTH = FRAME_LENGTH  # keep same as frame length for simplicity

# Parse MASK (frequency filter) env variable
# Helper function
def parse_mask(mask: str) -> dict:
    """
    Parse the MASK environment variable into a frequency mask config.

    Args:
        mask: Value of the MASK env variable.
            Supported formats:
                - nomask
                - bp<low>_<high>    (integer Hz)

            Constraints:
                - 0 <= low <= high <= NYQUIST_HZ

    Returns:
        Dict with mask configuration parameters.

    Raises:
        ValueError: If MASK format or frequency range is invalid.
    """
    if mask == "nomask":
        return {"mask": False}

    m = re.fullmatch(r"bp(\d+)_(\d+)", mask)
    if not m:
        raise ValueError(f'Invalid MASK="{mask}". Use MASK=nomask or MASK=bp<low>_<high> (0<=low<=high<={NYQUIST_HZ}).')

    low = int(m.group(1))
    high = int(m.group(2))

    if not (0 <= low <= high <= NYQUIST_HZ):
        raise ValueError(f'Invalid MASK="{mask}". Range must satisfy 0<=low<=high<={NYQUIST_HZ}.')

    return {"mask": True, "low_hz": float(low), "high_hz": float(high)}

cfg = parse_mask(MASK)

tag = "nomask" if not cfg.get("mask", False) else f"bp{int(cfg['low_hz'])}_{int(cfg['high_hz'])}"
OUT_DIR = TFRECORDS_ROOT / f"{MODEL}_{tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_CSV = SPLITS_DIR / f"{MODEL}.csv"

LABEL_MAP = {"neg": 0, "pos": 1}

# -----------------------
# TFRecord feature helpers
# -----------------------
def _bytes_feature(x: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[x]))

def _int64_feature(x: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(x)]))

def _float_feature(x: float) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(x)]))


# -----------------------
# Audio -> spectrogram
# -----------------------
def wav_to_logspec(wav_path: str) -> tf.Tensor:
    """
    Reads a WAV clip, forces SR to 4kHz (if 8kHz, downsample by 2),
    pads/trims to 4s, returns log-magnitude STFT spectrogram [T, F, 1].
    T is time frames, F is frequency bins.
    """
    wav_bytes = tf.io.read_file(wav_path)

    # Decode WAV to mono float32 samples in range [-1, 1];
    # returns audio with shape [N, 1] and sr (sample rate) as a scalar tensor.
    audio, sr = tf.audio.decode_wav(wav_bytes, desired_channels=1)
    
    # Remove single channel dimension ([N, 1] → [N])
    audio = tf.squeeze(audio, axis=-1)

    sr = tf.cast(sr, tf.int32)

    # Standardize sample rate to 4kHz:
    # - if 8kHz: take every other sample (2x downsample)
    # - if 4kHz: leave as-is
    if sr == 8000:
        audio = audio[::2]
        sr = 4000
    elif sr != 4000:
        raise ValueError(f"Unsupported sample rate {int(sr.numpy())} for {wav_path}")

    # Force fixed length (4s @ 4kHz)
    n = tf.shape(audio)[0]
    if n < EXPECTED_SAMPLES:
        audio = tf.pad(audio, [[0, EXPECTED_SAMPLES - n]])
    else:
        audio = audio[:EXPECTED_SAMPLES]

    # Compute short-time Fourier transform to obtain a time–frequency representation
    # of the audio signal (outputs complex numbers)
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FFT_LENGTH,
        pad_end=False,
    )

    # Convert complex STFT to magnitude spectrogram shape [T, F],
    # then apply log scaling to compress dynamic range for CNN stability
    mag = tf.abs(stft)
    mag = tf.math.log1p(mag) # log(1 + mag)

    # Optional frequency mask for spectrograms
    if cfg.get("mask", False):
        # Frequency bins for rfft: 0 to sr/2 with (FFT_LENGTH//2 + 1) bins
        freqs = tf.linspace(0.0, float(TARGET_SR) / 2.0, FFT_LENGTH // 2 + 1) # [F]
        
        # Create mask for frequencies to keep
        keep = tf.logical_and(freqs >= cfg["low_hz"], freqs <= cfg["high_hz"]) # [F]
        keep = tf.cast(keep, mag.dtype)

        # Apply the frequency mask across all time frames
        mag = mag * keep[tf.newaxis, :]

    mag = tf.expand_dims(mag, axis=-1)  # [T, F, 1]
    return mag


def serialize_example(spec: tf.Tensor, label: int, location: str, clip_rel: str) -> bytes:
    """
    Store the spectrogram as serialized tensor bytes + a few small metadata fields.
    """
    spec_bytes = tf.io.serialize_tensor(spec).numpy()

    features = {
        "spec": _bytes_feature(spec_bytes),
        "label": _int64_feature(label),
        "location": _bytes_feature(location.encode("utf-8")),
        "clip_wav_relpath": _bytes_feature(clip_rel.encode("utf-8")),
        # store params so it's obvious later what produced these TFRecords
        "target_sr": _int64_feature(TARGET_SR),
        "clip_len_s": _float_feature(CLIP_LEN_S),
        "frame_length": _int64_feature(FRAME_LENGTH),
        "frame_step": _int64_feature(FRAME_STEP),
        "fft_length": _int64_feature(FFT_LENGTH),
        "mask": _int64_feature(1 if cfg.get("mask", False) else 0),
        "low_hz": _float_feature(cfg.get("low_hz", -1.0)),
        "high_hz": _float_feature(cfg.get("high_hz", -1.0)),
    }

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


# -----------------------
# Main
# -----------------------
if not SPLITS_CSV.exists():
    raise FileNotFoundError(f"Missing split CSV: {SPLITS_CSV}")

df = pd.read_csv(SPLITS_CSV)
required = {"split", "label", "location", "clip_wav_relpath"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"{SPLITS_CSV} missing columns: {sorted(missing)}")

writers = {
    "train": tf.io.TFRecordWriter(str(OUT_DIR / "train.tfrecord")),
    "val": tf.io.TFRecordWriter(str(OUT_DIR / "val.tfrecord")),
    "test": tf.io.TFRecordWriter(str(OUT_DIR / "test.tfrecord")),
}

counts = {"train": 0, "val": 0, "test": 0}
skipped_missing = 0
skipped_bad_sr = 0

for _, row in df.iterrows():
    split = str(row["split"])
    if split not in writers:
        continue

    label_str = str(row["label"])
    if label_str not in LABEL_MAP:
        continue
    label = LABEL_MAP[label_str]

    location = str(row["location"])
    clip_rel = str(row["clip_wav_relpath"])
    clip_path = WAV_CLIPS_ROOT / clip_rel

    if not clip_path.exists():
        skipped_missing += 1
        continue

    try:
        spec = wav_to_logspec(str(clip_path))
    except ValueError:
        skipped_bad_sr += 1
        continue

    writers[split].write(serialize_example(spec, label, location, clip_rel))
    counts[split] += 1

for w in writers.values():
    w.close()

print(f"\nTFRecords written to: {OUT_DIR}")
print("Counts:", counts)
print("Skipped missing wav clips:", skipped_missing)
print("Skipped unsupported sample rate:", skipped_bad_sr)