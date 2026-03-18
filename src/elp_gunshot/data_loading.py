# src/elp_gunshot/data_loading.py
"""Shared TFRecord loading utilities for gunshot CNN training."""

from pathlib import Path

import tensorflow as tf

SPEC_SHAPE = (124, 129, 1)


def count_examples(path: str | Path) -> int:
    """Count examples in a TFRecord file."""
    return sum(1 for _ in tf.data.TFRecordDataset([str(path)]))


def parse_tfrecord_example(serialized, clip_id=False):
    """Parse a spectrogram TFRecord example into tensors expected by the CNN."""
    feature_desc = {
        "spec": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    if clip_id:
        feature_desc["clip_wav_relpath"] = tf.io.FixedLenFeature([], tf.string, default_value="")

    ex = tf.io.parse_single_example(serialized, feature_desc)

    spec = tf.io.parse_tensor(ex["spec"], out_type=tf.float32)
    spec = tf.reshape(spec, SPEC_SHAPE)

    label = tf.cast(ex["label"], tf.float32)
    label = tf.reshape(label, [1])

    if clip_id:
        return spec, label, ex["clip_wav_relpath"]
    return spec, label


def make_ds(path, parse_fn, batch_size, shuffle=False, drop_remainder=False):
    """Build a prefetching tf.data pipeline from a TFRecord file."""
    ds = tf.data.TFRecordDataset([str(path)], num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_class_weights(train_path: str | Path) -> dict:
    """Compute inverse-frequency class weights from training labels."""
    n0, n1 = 0, 0
    ds = tf.data.TFRecordDataset([str(train_path)])
    for serialized in ds:
        ex = tf.io.parse_single_example(serialized, {"label": tf.io.FixedLenFeature([], tf.int64)})
        if int(ex["label"].numpy()) == 1:
            n1 += 1
        else:
            n0 += 1

    total = n0 + n1
    if total == 0:
        raise ValueError(f"Cannot compute class weights: '{train_path}' contains no examples.")
    if n0 == 0 or n1 == 0:
        raise ValueError(
            f"Cannot compute class weights from '{train_path}': n0={n0}, n1={n1}. "
            "Both classes must be present."
        )

    return {0: total / (2.0 * n0), 1: total / (2.0 * n1)}
