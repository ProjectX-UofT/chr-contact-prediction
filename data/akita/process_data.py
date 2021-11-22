import json
import pathlib

import numpy as np
import tensorflow as tf
from natsort import natsorted

DATA_DIR = pathlib.Path(__file__).parent


def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type="ZLIB")


def parse_proto(example_protos):
    features = {
        "sequence": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }

    parsed_features = tf.io.parse_single_example(example_protos, features=features)
    seq = tf.io.decode_raw(parsed_features["sequence"], tf.uint8)
    tgts = tf.io.decode_raw(parsed_features["target"], tf.float16)
    return seq, tgts


def tfr_to_numpy(split, dummy=False):
    with open(DATA_DIR / "statistics.json", "r") as f:
        stats = json.load(f)
        n_seqs = stats[f"{split}_seqs"] if (not dummy) else 32
        seqs_shape = (n_seqs, stats["seq_length"])
        tgts_shape = (n_seqs, stats["target_length"], stats["num_targets"])

    with tf.name_scope('numpy'):
        raw_dir = DATA_DIR / "raw"
        tfr_paths = natsorted(raw_dir.glob(f"{split}-*.tfr"))
        tfr_paths = list(map(str, tfr_paths))
        assert tfr_paths

        # read TF Records
        dataset = tf.data.Dataset.from_tensor_slices(tfr_paths)
        dataset = dataset.flat_map(file_to_records)
        dataset = dataset.map(parse_proto)
        dataset = dataset.batch(1)

        print(f"Saving {n_seqs} {split} sequences...")
        seq_path = str(DATA_DIR / f"{split}_seqs.dat")
        tgt_path = str(DATA_DIR / f"{split}_targets.dat")

        seqs = np.memmap(seq_path, dtype=np.uint8, mode="w+", shape=seqs_shape)
        tgts = np.memmap(tgt_path, dtype=np.float16, mode="w+", shape=tgts_shape)

        for i, (seq_raw, tgt_raw) in enumerate(dataset):
            seqs[i] = seq_raw.numpy().reshape(seqs_shape[1], 4).argmax(axis=-1)
            tgts[i] = tgt_raw.numpy().reshape(tgts_shape[1:])
            if dummy and i == 31:
                break
        assert i + 1 == n_seqs

        seqs.flush()
        tgts.flush()


if __name__ == "__main__":
    dummy = True
    tfr_to_numpy(split="train", dummy=dummy)
    tfr_to_numpy(split="valid", dummy=dummy)
    tfr_to_numpy(split="test", dummy=dummy)
