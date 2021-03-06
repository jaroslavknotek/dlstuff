#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from mnist import MNIST

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, type=str, help="Learning decay rate type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer", default=200, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
if args.recodex:
    tf.keras.utils.get_custom_objects()["glorot_uniform"] = lambda: tf.keras.initializers.glorot_uniform(seed=42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load data
mnist = MNIST()

# Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[MNIST.H, MNIST.W, MNIST.C]),
    tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.relu),
    tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax),
])


def get_decay(decay, decay_steps, learning_rate, learning_rate_final):
    # If `args.decay` is set, then
    # - for `polynomial`, use `tf.keras.optimizers.schedules.PolynomialDecay`
    #   using the given `args.learning_rate_final`;
    # - for `exponential`, use `tf.keras.optimizers.schedules.ExponentialDecay`
    #   and set `decay_rate` appropriately to reach `args.learning_rate_final`
    #   just after the training (and keep the default `staircase=False`).

    if decay == "exponential":
        return tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            staircase=False,
            # kolikrat se to ma zmensovat abych nakonci docilil final
            decay_rate= learning_rate_final/learning_rate )
    elif decay == "polynomial":
        return tf.keras.optimizers.schedules.PolynomialDecay(
            learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=learning_rate_final)
    raise Exception("unknown decay type {}".format(decay))

    # For `SGD`, `args.momentum` can be specified. If `args.decay` is
    # not specified, pass the given `args.learning_rate` directly to the
    # optimizer.


lr_schedule = None
if args.decay is not None:
    # steps
    steps = args.epochs * ( mnist.train.size/args.batch_size  )
    lr_schedule = get_decay(args.decay, steps, args.learning_rate, args.learning_rate_final)
else:
    lr_schedule = args.learning_rate


# In both cases, `decay_steps` should be total number of training batches.
# If a learning rate schedule is used, you can find out current learning rate
# by using `model.optimizer.learning_rate(model.optimizer.iterations)`,
# so after training this value should be `args.learning_rate_final`.
def get_optimizer(optimizer,lr_schedule, momentum):
    sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    if momentum is not None:
        sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)
    return {
        "SGD": sgd,
        "Adam": tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                         amsgrad=False)
    }.get(optimizer, None)

# TODO: Use the required `args.optimizer` (either `SGD` or `Adam`).
optimizer = get_optimizer(args.optimizer, lr_schedule, args.momentum)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1, update_freq=1000, profile_batch=1)
tb_callback.on_train_end = lambda *_: None
model.fit(
    mnist.train.data["images"], mnist.train.data["labels"],
    batch_size=args.batch_size, epochs=args.epochs,
    validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    callbacks=[tb_callback],
)

test_logs = model.evaluate(
    mnist.test.data["images"], mnist.test.data["labels"], batch_size=args.batch_size,
)
tb_callback.on_epoch_end(1,
                         dict(("val_test_" + metric, value) for metric, value in zip(model.metrics_names, test_logs)))
# TODO asses the final learning state is the same as the model's
# final_learning_rate = model.optimizer.learning_rate(model.optimizer.iterations)
# print(final_learning_rate, args.learning_rate_final)
# TODO: Write test accuracy as percentages rounded to two decimal places.
accuracy = test_logs[1]
with open("mnist_training.out", "w") as out_file:
    print("{:.2f}".format(100 * accuracy), file=out_file)
