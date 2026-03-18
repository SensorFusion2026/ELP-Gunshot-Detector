# src/elp_gunshot/train_cnn.py
# Usage examples:
#   python -m elp_gunshot.train_cnn
#   MODEL=model2 TAG=bp100_1800 python -m elp_gunshot.train_cnn
#
# Notes:
# - Expects TFRecords at: data/tfrecords/<MODEL>_<TAG>/{train,val,test}.tfrecord
# - TFRecord schema: "spec" (serialized tensor), "label" (int64)

import os
import json
import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from elp_gunshot.config.paths import RUNS_DIR as DEFAULT_RUNS_DIR, TFRECORDS_ROOT

# =========================
# GPU CONFIG
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU detected. Memory growth enabled.")
    except RuntimeError as e:
        print("Error setting memory growth:", e)
else:
    print("No GPU detected, running on CPU.")

# Enable mixed precision for Tesla T4
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision enabled.")

# =========================
# CONFIG
# =========================
MODEL = os.getenv("MODEL", "model1")    # model1 / model2 / model3
TAG   = os.getenv("TAG", "nomask")      # nomask / bp100_1800 / etc

BASE_DIR = TFRECORDS_ROOT / f"{MODEL}_{TAG}"

TRAIN_TF = BASE_DIR / "train.tfrecord"
VAL_TF = BASE_DIR / "val.tfrecord"
TEST_TF  = BASE_DIR / "test.tfrecord"

RUNS_DIR = Path(os.getenv("RUNS_DIR", str(DEFAULT_RUNS_DIR)))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
#BATCH_SIZE = 64 #for best model 3 training
EPOCHS = int(os.getenv("EPOCHS", 10))
#EPOCHS = 30 #for best model 2 training
#EPOCHS = 40 #for best model 3 training 
LEARNING_RATE = 1e-4
#LEARNING_RATE = 3e-5 #for best model 3 training 
AUTOTUNE = tf.data.AUTOTUNE

SPEC_HEIGHT = 124
SPEC_WIDTH = 129

if not TRAIN_TF.exists() or not VAL_TF.exists() or not TEST_TF.exists():
    raise FileNotFoundError(
        f"Missing TFRecords under {BASE_DIR}\n"
        f"Expected:\n"
        f"  {TRAIN_TF}\n  {VAL_TF}\n  {TEST_TF}\n"
    )

# Create a run name that identifies what produced it
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"{MODEL}_{TAG}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_e{EPOCHS}_{ts}"
EXPORT_DIR = RUNS_DIR / run_name
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nTFRecords:   {BASE_DIR}")
print(f"Export dir:  {EXPORT_DIR}\n")

# =========================
# TFRecord schema
# =========================
feature_description = {
    "spec": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "clip_wav_relpath": tf.io.FixedLenFeature([], tf.string),
}

# =========================
# TFRecord parser
# =========================
def parse_tfrecord(example, return_id=False):
    parsed = tf.io.parse_single_example(example, feature_description)

    spec = tf.io.parse_tensor(parsed["spec"], out_type=tf.float32)
    # Per-example min-max normalization (simple baseline)
    spec = (spec - tf.reduce_min(spec)) / (tf.reduce_max(spec) - tf.reduce_min(spec) + 1e-8)

    if tf.rank(spec) == 2:
        spec = tf.expand_dims(spec, axis=-1)
    spec.set_shape([SPEC_HEIGHT, SPEC_WIDTH, 1])

    label = tf.cast(parsed["label"], tf.float32)
    label = tf.reshape(label, (1,))

    if return_id:
        return spec, label, parsed["clip_wav_relpath"]
    return spec, label

# =========================
# Dataset loader
# =========================
def load_dataset(tfrecord_path, shuffle=False, repeat=False, drop_remainder=True, return_id=False):
    ds = tf.data.TFRecordDataset(str(tfrecord_path), num_parallel_reads=AUTOTUNE)
    ds = ds.map(
        lambda ex: parse_tfrecord(ex, return_id=return_id),
        num_parallel_calls=AUTOTUNE,
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds

# =========================
# Count number of examples
# =========================
def count_examples(tfrecord_path):
    n = 0
    for _ in tf.data.TFRecordDataset(str(tfrecord_path)):
        n += 1
    return n

# =========================
# Datasets
# =========================
train_ds = load_dataset(TRAIN_TF, shuffle=True, repeat=False)
val_ds = load_dataset(VAL_TF, shuffle=False, repeat=False)
test_ds  = load_dataset(TEST_TF, shuffle=False, repeat=False, drop_remainder=False, return_id=True) # Keep last batch

n_train = count_examples(TRAIN_TF)
n_val   = count_examples(VAL_TF)
n_test  = count_examples(TEST_TF)

train_steps = math.ceil(n_train / BATCH_SIZE)
val_steps   = math.ceil(n_val / BATCH_SIZE)

print(f"Examples: train={n_train}, val={n_val}, test={n_test}")
print(f"Steps:    train={train_steps}, val={val_steps}\n")

# =========================
# Quick label check
# =========================
print("Sample labels from training dataset:")
for batch_spec, batch_label in train_ds.take(1):
    print(batch_label.numpy().flatten())

# =========================
# Compute class weights
# =========================
def get_class_weights_from_tfrecord(tfrecord_path):
    ds = tf.data.TFRecordDataset(str(tfrecord_path))
    ds = ds.map(lambda ex: tf.io.parse_single_example(ex, {"label": tf.io.FixedLenFeature([], tf.int64)}))
    labels = []
    for ex in ds:
        labels.append(int(ex["label"].numpy()))
    counts = np.bincount(labels, minlength=2)
    total = counts.sum()
    if counts[0] == 0 or counts[1] == 0:
        return {0: 1.0, 1: 1.0}
    return {0: total / (2 * counts[0]), 1: total / (2 * counts[1])}

class_weight = get_class_weights_from_tfrecord(TRAIN_TF)
print("Class weights:", class_weight, "\n")

# =========================
# CNN MODEL
# =========================
model = models.Sequential([
    layers.Input(shape=(SPEC_HEIGHT, SPEC_WIDTH, 1)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2),

    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid", dtype='float32')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

# =========================
# Callbacks (organized logging)
# =========================
ckpt_path = EXPORT_DIR / "best_model.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_path),
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(str(EXPORT_DIR / "history.csv"), append=False),
    tf.keras.callbacks.TensorBoard(log_dir=str(EXPORT_DIR / "logs")),
]

# =========================
# Save run params up front
# =========================
run_params = {
    "model": MODEL,
    "tag": TAG,
    "tfrecord_dir": str(BASE_DIR),
    "train_tfrecord": str(TRAIN_TF),
    "val_tfrecord": str(VAL_TF),
    "test_tfrecord": str(TEST_TF),
    "spec_shape": [SPEC_HEIGHT, SPEC_WIDTH, 1],
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "class_weight": class_weight,
    "mixed_precision": True,
}
with open(EXPORT_DIR / "params.json", "w") as f:
    json.dump(run_params, f, indent=2)

# =========================
# Train
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    class_weight=class_weight,
    callbacks=callbacks,
)

# =========================
# Test evaluation + save predictions
# =========================
print("\nEvaluating best model (from checkpoint) on TEST set...")

model = tf.keras.models.load_model(str(ckpt_path))

# Collect predictions and true labels
clip_ids = []
y_true, y_pred, y_score = [], [], []

for batch_spec, batch_label, batch_clip_rel in test_ds:
    probs = model.predict(batch_spec, verbose=0).flatten()
    preds = (probs > 0.5).astype(int)

    y_score.extend(probs.tolist())
    y_pred.extend(preds.tolist())
    y_true.extend(batch_label.numpy().flatten().astype(int).tolist())

    clip_ids.extend([x.decode("utf-8") for x in batch_clip_rel.numpy()])

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

assert len(clip_ids) == len(y_true), f"Mismatch: clip_ids={len(clip_ids)} y_true={len(y_true)}"

# Compute metrics
try:
    auc = float(roc_auc_score(y_true, y_score))
except ValueError:
    auc = float("nan")

test_metrics = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    "auc": auc,
    "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    "num_test_examples": int(len(y_true)),
}


with open(EXPORT_DIR / "test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2)

with open(EXPORT_DIR / "test_predictions.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["clip_wav_relpath", "y_true", "y_pred", "y_score"])
    for cid, yt, yp, ys in zip(clip_ids, y_true, y_pred, y_score):
        w.writerow([cid, int(yt), int(yp), float(ys)])

print("Test metrics:")
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

# =========================
# Save final model
# =========================
final_model_path = EXPORT_DIR / "final_model.keras"
model.save(str(final_model_path))
print("\nSaved:")
print("  params:          ", EXPORT_DIR / "params.json")
print("  history:         ", EXPORT_DIR / "history.csv")
print("  best_model:      ", ckpt_path)
print("  test_metrics:    ", EXPORT_DIR / "test_metrics.json")
print("  test_predictions:", EXPORT_DIR / "test_predictions.csv")
print("  final_model:     ", final_model_path)
