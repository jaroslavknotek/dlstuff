#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=500, type=int,
                    help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=100, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layers", default="500", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=11, type=int, help="Window size to use.")
args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.


hyper = {
    "l2": 0.001,  # 0, 0.001, 0.0001, 0.00001,
    "dropout": 0.2,
    "ls": 0.001,
    "samples": 1000
}

reg = tf.keras.regularizers.L1L2(0, hyper["l2"])

# # Create the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32))
model.add(tf.keras.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(hyper["dropout"]))
for hidden_layer in args.hidden_layers:
    model.add(tf.keras.layers.Dense(hidden_layer, activation=tf.nn.relu, kernel_regularizer=reg, bias_regularizer=reg))
    model.add(tf.keras.layers.Dropout(hyper["dropout"]))
model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

loss = None
metric = None

if hyper["ls"] > 0:
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=hyper["ls"])
    metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
    train_labels = tf.keras.utils.to_categorical(uppercase_data.train.data["labels"])
    dev_labels = tf.keras.utils.to_categorical(uppercase_data.dev.data["labels"])
    test_labels = tf.keras.utils.to_categorical(uppercase_data.test.data["labels"])
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    train_labels = uppercase_data.train.data["labels"]
    dev_labels = uppercase_data.dev.data["labels"]
    test_labels = uppercase_data.test.data["labels"]

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=loss,
    metrics=[metric],
)

tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None

model.fit(
    uppercase_data.train.data["windows"][:hyper["samples"]], train_labels[:hyper["samples"]],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(uppercase_data.dev.data["windows"], dev_labels),
    callbacks=[tb_callback],
)
#
test_logs = model.evaluate(uppercase_data.test.data["windows"], test_labels, batch_size=args.batch_size)
tb_callback.on_epoch_end(1,
                         dict(("val_test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))

accuracy = test_logs[model.metrics_names.index("accuracy")]

# model.save('path_to_my_model.h5')

res = model.predict(uppercase_data.test.data["windows"])
reference_file_content = None
with open("uppercase_data_test.txt", "r", encoding="utf-8") as reference_file:
    reference_file_content = reference_file.read()

table = dict(enumerate(uppercase_data.test.alphabet))

out_content = []
print(len(res), len(uppercase_data.test.data["windows"]))
for pred, window, i in zip(res, uppercase_data.test.data["windows"], range(len(res))):
    number = window[args.window]
    letter = table.get(number, reference_file_content[i])
    if pred[1] > pred[0]:
        letter = letter.capitalize()
    out_content.append(letter)

with open("uppercase_test.txt", "w", encoding="utf-8") as out_file:
    out_file.write(''.join(out_content))
