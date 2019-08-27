from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import runners


# Shared flags
tf.app.flags.DEFINE_enum('mode', 'train',
                         ['train', 'eval'],
                         "The mode of the binary.")
tf.app.flags.DEFINE_enum('model', 'gmvae',
                         ['gmvae', 'vae', 'vae_gmp'],
                         "Model choice.")
tf.app.flags.DEFINE_integer('latent_size', 8,
                            "Number of dimensions in the latent state.")
tf.app.flags.DEFINE_integer('hidden_size', 64,
                            "Number of dimensions in the hidden layers.")
tf.app.flags.DEFINE_integer('num_layers', 1,
                            "Number of hidden layers in the internal networks.")
tf.app.flags.DEFINE_integer('mixture_components', 10,
                            "Number of mixture components if using either a GMVAE "
                            "or a standard VAE with a GM prior.")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            "Batch size.")
tf.app.flags.DEFINE_string('logdir', '/tmp/smc_vi',
                           "The directory to keep checkpoints and summaries in.")
tf.app.flags.DEFINE_integer('random_seed', None,
                            "A random seed for seeding the TensorFlow graph.")


# Training/Evaluation flags
tf.app.flags.DEFINE_float('learning_rate', 0.001,
                          "The learning rate for ADAM.")
tf.app.flags.DEFINE_integer('max_steps', int(1e9),
                            "The number of gradient update steps to train for.")
tf.app.flags.DEFINE_integer('num_samples', 100,
                            "Number of samples for generated images.")
tf.app.flags.DEFINE_integer('early_stop_rounds', 1000,
                            "Number of steps before terminating training due to early stopping.")
tf.app.flags.DEFINE_float('early_stop_threshold', 0.001,
                          "Early stopping threshold for percentage improvement in validation loss.")
tf.app.flags.DEFINE_integer('summarise_every', 50,
                            "The number of steps between summaries.")
tf.app.flags.DEFINE_string('gpu_id', '0',
                            "GPU device id to use.")
tf.app.flags.DEFINE_string('gpu_num', '0',
                            "Comma-separated list of GPU ids to use.")
tf.app.flags.DEFINE_enum('split', 'train',
                         ['train', 'test'],
                         "Split to evaluate the model on.")


FLAGS = tf.app.flags.FLAGS


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if FLAGS.mode == 'train':
        runners.run_train(FLAGS)
    elif FLAGS.mode == 'eval':
        runners.run_eval(FLAGS)


if __name__ == '__main__':
    tf.compat.v1.app.run(main)