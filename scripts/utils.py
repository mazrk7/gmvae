from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

import tensorflow as tf

from sklearn.manifold import TSNE


class EarlyStoppingHook(tf.estimator.SessionRunHook):
    """Monitor to request stop when 'loss_op' stops increasing."""

    def __init__(self, loss_op, max_steps=100, threshold=0.001):
        self._loss_op = loss_op
        self._max_steps = max_steps
        self._threshold = threshold
        self._last_step = -1

        # Records the number of steps for which the loss has been non-increasing
        self._steps = 0
        self._prev_loss = None


    def before_run(self, run_context):
        return tf.compat.v1.train.SessionRunArgs(
            {'global_step': tf.compat.v1.train.get_or_create_global_step(),
             'current_loss': self._loss_op})
              

    def after_run(self, run_context, run_values):
        curr_loss = run_values.results['current_loss']
        curr_step = run_values.results['global_step']
        self._steps += 1

        # Guard against the global step going backwards e.g. during recovery
        if self._last_step == -1 or self._last_step > curr_step:
            tf.compat.v1.logging.info("EarlyStoppingHook resetting last_step.")
            self._last_step = curr_step
            self._steps = 0
            self._prev_loss = None

            return

        self._last_step = curr_step
        # If no previous loss or current loss is decreasing as desired
        if (self._prev_loss is None or curr_loss < 
            (self._prev_loss - self._prev_loss * self._threshold)):
            self._prev_loss = curr_loss
            self._steps = 0

        # If early stopping condition has been met
        if self._steps >= self._max_steps:
            tf.compat.v1.logging.info("[Early Stopping Criterion Satisfied]")
            run_context.request_stop()


# Creates a logging hook that prints the loss values periodically
def create_logging_hook(step, loss, every_steps=50):

    def summary_formatter(log_dict):
        return "Step %d, %s: %f" % (log_dict['step'], 
            'loss', log_dict['loss'])

    logging_hook = tf.compat.v1.train.LoggingTensorHook(
        {'step': step, 
         'loss': loss},
        every_n_iter=every_steps,
        formatter=summary_formatter)

    return logging_hook


def restore_checkpoint_if_exists(saver, sess, logdir):
    """Looks for a checkpoint and restores the session if found.

    Args:
        saver: A tf.compat.v1.train.Saver for restoring the session.
        sess: A TensorFlow session.
        logdir: The directory to look for checkpoints in.

    Returns:
        True if a checkpoint was found and restored, False otherwise.
    """

    checkpoint = tf.compat.v1.train.get_checkpoint_state(logdir)

    if checkpoint:
        checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
        full_checkpoint_path = os.path.join(logdir, checkpoint_name)
        saver.restore(sess, full_checkpoint_path)

        return True

    return False


def wait_for_checkpoint(saver, sess, logdir):
    """Loops until the session is restored from a checkpoint in logdir.

    Args:
        saver: A tf.compat.v1.train.Saver for restoring the session.
        sess: A TensorFlow session.
        logdir: The directory to look for checkpoints in.
    """

    while not restore_checkpoint_if_exists(saver, sess, logdir):
        tf.compat.v1.logging.info("Checkpoint not found in %s, sleeping for 60 seconds." % logdir)
        time.sleep(60)


def pack_images(images, rows, cols):
    """Helper utility to make a field of images."""

    shape = tf.shape(input=images)
    width = shape[-3]
    height = shape[-2]
    depth = shape[-1]

    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(input=images)[0]

    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)

    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.reshape(images, [1, rows * width, cols * height, depth])

    return images


def image_tile_summary(name, tensor, rows=8, cols=8):
    tf.compat.v1.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def flatten_tensor(inputs, shape, name='flattened'):
    return tf.reshape(inputs, [-1, tf.reduce_prod(shape)], name=name)


def unflatten_tensor(inputs, shape, name='unflattened'):    
    return tf.reshape(inputs, [-1, shape[0], shape[1], shape[2]], name=name)


def reduce_dimensionality(data, dim=2, perplexity=40):
    if(data.shape[-1] > 2):
        tsne = TSNE(n_components=dim, verbose=1, perplexity=perplexity, n_iter=300)
        data = tsne.fit_transform(data)

    return data


def mode_tensor(x):
    """Computes the mode of the float Tensor x."""

    y, idx, count = tf.unique_with_counts(x)
    mode = y[tf.argmax(count)]

    return tf.cast(mode, dtype=tf.float32)


def entropy(logits, targets):
    """Computes entropy as -sum(targets*log(predicted))"""

    log_q = tf.nn.log_softmax(logits)

    return -tf.reduce_sum(targets*log_q, axis=1)


def cluster_acc(logits, labels, no_components):
    """Computes the clustering accuracy metric."""

    cat_preds = tf.argmax(logits, axis=1)
    real_preds = tf.zeros(tf.shape(cat_preds))

    for k in xrange(no_components):
        idx = tf.equal(cat_preds, k)
        lab = tf.boolean_mask(labels, idx)

        modes = tf.cond(tf.equal(tf.size(lab), 0),
            lambda: 0.0,
            lambda: mode_tensor(lab))

        real_preds += tf.cast(idx, dtype=tf.float32) * modes

    matches = tf.equal(real_preds, tf.cast(labels, dtype=tf.float32))

    return tf.reduce_mean(tf.cast(matches, dtype=tf.float32))