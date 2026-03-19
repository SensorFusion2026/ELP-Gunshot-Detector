# src/elp_gunshot/train_cnn.py
# Usage examples:
#   python -m elp_gunshot.train_cnn
#   MODEL=model2 TAG=bp100_1800 python -m elp_gunshot.train_cnn
#
# Notes:
# - Expects TFRecords at: data/tfrecords/<MODEL>_<TAG>/{train,val,test}.tfrecord
# - TFRecord schema: "spec" (serialized normalized tensor), "label" (int64)

import csv
import functools
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from elp_gunshot.cnn import CNN
from elp_gunshot.config.paths import RUNS_DIR as DEFAULT_RUNS_DIR, TFRECORDS_ROOT
from elp_gunshot.data_loading import (
    SPEC_SHAPE,
    count_examples,
    get_class_weights,
    make_ds,
    parse_tfrecord_example,
)

_BCE_LOSS = tf.keras.losses.BinaryCrossentropy()

def _configure_gpu() -> bool:
    """Enable GPU-friendly settings and return whether mixed precision is enabled."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[train_cnn] No GPU detected; running on CPU.")
        return False

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"[train_cnn] GPU detected: {[g.name for g in gpus]} - mixed precision enabled.")
        return True
    except RuntimeError as err:
        print(f"[train_cnn] Could not enable GPU memory growth: {err}")
        return False


def _save_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _validate_artifacts(export_dir: Path, required_names: list[str]) -> None:
    missing = [name for name in required_names if not (export_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Run completed with missing artifacts in {export_dir}: {missing}. "
            "Inspect training logs for failures."
        )


def _collect_predictions(model, dataset, has_clip_id: bool):
    """
    Collect labels, probabilities, predictions-ready metadata from a dataset.
    Returns:
      clip_ids: list[str] | None
      y_true_arr: np.ndarray
      y_score_arr: np.ndarray
    """
    clip_ids = [] if has_clip_id else None
    y_true, y_score = [], []

    for batch in dataset:
        if has_clip_id:
            x_batch, y_batch, id_batch = batch
        else:
            x_batch, y_batch = batch

        probs = model(x_batch, training=False).numpy().flatten()
        y_score.extend(probs.tolist())
        y_true.extend(y_batch.numpy().flatten().astype(int).tolist())

        if has_clip_id:
            clip_ids.extend([cid.decode("utf-8") for cid in id_batch.numpy()])

    return clip_ids, np.array(y_true), np.array(y_score)


def _confusion_dict(y_true_arr: np.ndarray, y_pred_arr: np.ndarray) -> dict:
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _compute_metrics(y_true_arr: np.ndarray, y_score_arr: np.ndarray, threshold: float) -> dict:
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    try:
        auc = float(roc_auc_score(y_true_arr, y_score_arr))
    except ValueError:
        auc = float("nan")

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "bce_loss": float(_BCE_LOSS(y_true_arr, y_score_arr).numpy()),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "auc": auc,
        "confusion_matrix": _confusion_dict(y_true_arr, y_pred_arr),
        "n_examples": int(len(y_true_arr)),
    }

    p = metrics["precision"]
    r = metrics["recall"]
    metrics["f1"] = float(0.0 if (p + r) == 0 else (2 * p * r) / (p + r))
    return metrics


def _choose_threshold_from_validation(
    y_true_arr: np.ndarray,
    y_score_arr: np.ndarray,
    min_precision: float,
) -> tuple[float, list[dict]]:
    """
    Choose threshold on validation set.

    Rule:
    - among thresholds with precision >= min_precision, maximize recall
    - tie-break by F1
    - if none satisfy precision floor, maximize F1
    """

    candidate_rows = []
    threshold_values = np.linspace(0.0, 1.0, num=101)

    for threshold in threshold_values:
        metrics = _compute_metrics(y_true_arr, y_score_arr, float(threshold))
        candidate_rows.append(metrics)

    valid = [row for row in candidate_rows if row["precision"] >= min_precision]

    if valid:
        best = max(valid, key=lambda row: (row["recall"], row["f1"], -abs(row["threshold"] - 0.5)),)
    else:
        best = max(candidate_rows, key=lambda row: (row["f1"], row["recall"]))

    return float(best["threshold"]), candidate_rows


def _write_predictions_csv(
    path: Path,
    clip_ids,
    y_true_arr: np.ndarray,
    y_score_arr: np.ndarray,
    threshold: float,
) -> None:
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_wav_relpath", "y_true", "y_pred", "y_score", "threshold"])
        if clip_ids is None:
            for idx, (yt, yp, ys) in enumerate(zip(y_true_arr, y_pred_arr, y_score_arr)):
                writer.writerow([f"example_{idx}", int(yt), int(yp), f"{float(ys):.6f}", f"{threshold:.6f}"])
        else:
            for cid, yt, yp, ys in zip(clip_ids, y_true_arr, y_pred_arr, y_score_arr):
                writer.writerow([cid, int(yt), int(yp), f"{float(ys):.6f}", f"{threshold:.6f}"])


def _write_threshold_table(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["threshold", "accuracy", "precision", "recall", "f1", "auc", "tp", "tn", "fp", "fn", "n_examples"]
        )
        for row in sorted(rows, key=lambda r: r["threshold"]):
            cm = row["confusion_matrix"]
            writer.writerow(
                [
                    f"{row['threshold']:.6f}",
                    f"{row['accuracy']:.6f}",
                    f"{row['precision']:.6f}",
                    f"{row['recall']:.6f}",
                    f"{row['f1']:.6f}",
                    f"{row['auc']:.6f}" if not np.isnan(row["auc"]) else "nan",
                    cm["tp"],
                    cm["tn"],
                    cm["fp"],
                    cm["fn"],
                    row["n_examples"],
                ]
            )


def main():
    mixed_precision_enabled = _configure_gpu()

    # Model3 best defaults, while keeping env overrides for experiments.
    model_name = os.getenv("MODEL", "model3")
    tag = os.getenv("TAG", "nomask")
    batch_size = int(os.getenv("BATCH_SIZE", 64))
    epochs = int(os.getenv("EPOCHS", 40))
    learning_rate = float(os.getenv("LEARNING_RATE", "3e-5"))
    dropout_rate = float(os.getenv("DROPOUT_RATE", "0.5"))
    early_stop_patience = int(os.getenv("EARLY_STOP_PATIENCE", 8))
    min_precision = float(os.getenv("VAL_MIN_PRECISION", "0.75"))
    seed = int(os.getenv("SEED", "42"))

    tf.keras.utils.set_random_seed(seed)

    base_dir = TFRECORDS_ROOT / f"{model_name}_{tag}"
    train_tf = base_dir / "train.tfrecord"
    val_tf = base_dir / "val.tfrecord"
    test_tf = base_dir / "test.tfrecord"
    metadata_path = base_dir / "metadata.json"

    for tfrecord_path in (train_tf, val_tf, test_tf):
        if not tfrecord_path.exists():
            raise FileNotFoundError(
                f"Missing TFRecord: {tfrecord_path}. "
                f"Expected dataset root: {base_dir}"
            )

    runs_dir = Path(os.getenv("RUNS_DIR", str(DEFAULT_RUNS_DIR)))
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{tag}_bs{batch_size}_lr{learning_rate}_e{epochs}_{ts}"
    export_dir = runs_dir / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[train_cnn] TFRecords:  {base_dir}")
    print(f"[train_cnn] Export dir: {export_dir}\n")

    parse_fn = functools.partial(parse_tfrecord_example, clip_id=False)
    parse_with_id_fn = functools.partial(parse_tfrecord_example, clip_id=True)

    train_ds = make_ds(train_tf, parse_fn, batch_size, shuffle=True, drop_remainder=False)
    val_ds = make_ds(val_tf, parse_fn, batch_size, shuffle=False, drop_remainder=False)
    val_ds_with_id = make_ds(val_tf, parse_with_id_fn, batch_size, shuffle=False, drop_remainder=False)
    test_ds = make_ds(test_tf, parse_with_id_fn, batch_size, shuffle=False, drop_remainder=False)

    n_train = count_examples(train_tf)
    n_val = count_examples(val_tf)
    n_test = count_examples(test_tf)
    print(f"[train_cnn] Examples: train={n_train}, val={n_val}, test={n_test}")

    print("[train_cnn] Sample labels from one training batch:")
    for _, batch_label in train_ds.take(1):
        print(batch_label.numpy().flatten())

    class_weights = get_class_weights(train_tf)
    print(f"[train_cnn] class_weights = {class_weights}\n")

    params = {
        "model": model_name,
        "tag": tag,
        "tfrecord_dir": str(base_dir),
        "train_tfrecord": str(train_tf),
        "val_tfrecord": str(val_tf),
        "test_tfrecord": str(test_tf),
        "tfrecord_metadata": str(metadata_path) if metadata_path.exists() else None,
        "input_shape": list(SPEC_SHAPE),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "dropout_rate": dropout_rate,
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "mixed_precision": mixed_precision_enabled,
        "early_stop_patience": early_stop_patience,
        "val_min_precision": min_precision,
        "seed": seed,
    }

    if metadata_path.exists():
        params["normalization"] = json.loads(metadata_path.read_text())

    _save_json(export_dir / "params.json", params)

    model = CNN(input_shape=SPEC_SHAPE, dropout_rate=dropout_rate)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    model.build((None, *SPEC_SHAPE))
    model.summary()

    ckpt_path = export_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(export_dir / "history.csv"), append=False),
        tf.keras.callbacks.TensorBoard(log_dir=str(export_dir / "logs")),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    final_model_path = export_dir / "final_model.keras"
    model.save(str(final_model_path))

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Expected best checkpoint not found: {ckpt_path}. "
            "ModelCheckpoint may have failed before first validation pass."
        )

    best_model = tf.keras.models.load_model(str(ckpt_path))

    print("\n[train_cnn] Running validation inference for threshold selection...")
    val_clip_ids, val_y_true, val_y_score = _collect_predictions(best_model, val_ds_with_id, has_clip_id=True)

    chosen_threshold, val_threshold_rows = _choose_threshold_from_validation(
        val_y_true,
        val_y_score,
        min_precision=min_precision,
    )
    val_metrics = _compute_metrics(val_y_true, val_y_score, chosen_threshold)

    print(
        "[train_cnn] Chosen validation threshold = "
        f"{chosen_threshold:.6f} "
        f"(precision={val_metrics['precision']:.4f}, "
        f"recall={val_metrics['recall']:.4f}, "
        f"f1={val_metrics['f1']:.4f})"
    )

    _save_json(export_dir / "val_metrics.json", val_metrics)
    _write_predictions_csv(
        export_dir / "val_predictions.csv",
        val_clip_ids,
        val_y_true,
        val_y_score,
        chosen_threshold,
    )
    _write_threshold_table(export_dir / "val_metrics_by_threshold.csv", val_threshold_rows)

    print("\n[train_cnn] Evaluating best model on test split...")
    test_clip_ids, test_y_true, test_y_score = _collect_predictions(best_model, test_ds, has_clip_id=True)
    test_metrics = _compute_metrics(test_y_true, test_y_score, chosen_threshold)
    _save_json(export_dir / "test_metrics.json", test_metrics)

    _write_predictions_csv(
        export_dir / "test_predictions.csv",
        test_clip_ids,
        test_y_true,
        test_y_score,
        chosen_threshold,
    )

    _validate_artifacts(
        export_dir,
        [
            "params.json",
            "history.csv",
            "best_model.keras",
            "final_model.keras",
            "val_metrics.json",
            "val_predictions.csv",
            "val_metrics_by_threshold.csv",
            "test_metrics.json",
            "test_predictions.csv",
            "logs",
        ],
    )

    print(f"\n[train_cnn] Run complete: {export_dir}")
    for artifact in (
        "params.json",
        "history.csv",
        "best_model.keras",
        "final_model.keras",
        "val_metrics.json",
        "val_predictions.csv",
        "val_metrics_by_threshold.csv",
        "test_metrics.json",
        "test_predictions.csv",
        "logs/",
    ):
        print(f"  {artifact:<24} ✓")

    print("\nValidation metrics:")
    for key, value in val_metrics.items():
        print(f"  {key}: {value}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()