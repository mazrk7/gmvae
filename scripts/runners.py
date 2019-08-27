from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

import helpers
import gmvae
import vae


def reduce_dimensionality(data, dim=2, perplexity=40):
    if(data.shape[-1] > 2):
        tsne = TSNE(n_components=dim, verbose=1, perplexity=perplexity, n_iter=300)
        data = tsne.fit_transform(data)

    return data


# Creates a logging hook that prints the loss values periodically
def create_logging_hook(step, train_loss, test_loss, every_steps=50):

    def summary_formatter(log_dict):
        return "Step %d, %s: %f, %s: %f" % (log_dict['step'], 
            'train_loss', log_dict['train_loss'],
            'test_loss', log_dict['test_loss'])

    logging_hook = tf.compat.v1.train.LoggingTensorHook(
        {'step': step, 
         'train_loss': train_loss,
         'test_loss': test_loss},
        every_n_iter=every_steps,
        formatter=summary_formatter)

    return logging_hook


def create_dataset(config, split, shuffle, repeat):
    """Creates the MNIST dataset for a given config.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object. This function expects the properties
            batch_size, dataset_path, and latent_size to be defined.
        split: The dataset split to load.
        shuffle: If true, shuffle the dataset randomly.
        repeat: If true, repeat the dataset endlessly.

    Returns:
        images: A batch of image sequences represented as a dense Tensor 
            of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1].
        image_shape: A shape Tensor for the images contained in the dataset.
        labels: A batch of integer labels, for use in evaluation of clustering.
    """

    dataset, datasets_info = tfds.load(name='mnist',
                                      split=split,
                                      with_info=True,
                                      as_supervised=False)

    image_shape = datasets_info.features['image'].shape

    def _preprocess(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval
        return image, sample['label']


    dataset = (dataset.map(_preprocess)
                      .batch(config.batch_size)
                      .prefetch(tf.data.experimental.AUTOTUNE))
    
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(datasets_info.splits[split].num_examples)

    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    images, labels = iterator.get_next()
    
    iterator_init = iterator.initializer

    return images, labels, iterator_init


def create_model(config, split, inputs, labels, shape):
    """Creates the model

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object. This function expects the properties
            batch_size, dataset_path, and latent_size to be defined.
        split: The dataset split to load.
        inputs: A batch of image sequences.
        labels: A batch of integer labels, for use in evaluation of clustering.
        shape: A shape tensor used to re-shape the inputs to feed into the model.

    Returns:
        model: Either a gmvae.TrainableGMVAE or vae.TrainableVAE model object.
        loss: A float Tensor containing the model's loss value.
    """

    flat_inputs = helpers.flatten_tensor(inputs, shape, name=split + '_inputs')
    data_dim = flat_inputs.get_shape().as_list()[1]

    if config.model == 'gmvae':
        # Create a gmvae.TrainableGMVAE model object
        model = gmvae.create_gmvae(data_dim,
                                   config.latent_size,
                                   mixture_components=config.mixture_components,
                                   fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                                   sigma_min=0.0,
                                   raw_sigma_bias=0.5,
                                   temperature=0.5)
    elif config.model == 'vae_gmp':
        # Create a mixture prior vae.TrainableVAE model object
        model = vae.create_vae(data_dim,
                               config.latent_size,
                               mixture_components=config.mixture_components,
                               fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                               sigma_min=0.0,
                               raw_sigma_bias=0.5)
    else:
        # Create a standard vae.TrainableVAE model object
        model = vae.create_vae(data_dim,
                               config.latent_size,
                               fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                               sigma_min=0.0,
                               raw_sigma_bias=0.5)

    with tf.name_scope(split):
        if config.model == 'gmvae':
            loss = model.run_model(flat_inputs, flat_inputs, labels)
        else:
            loss = model.run_model(flat_inputs, flat_inputs)

        with tf.name_scope('image_summaries'):
            helpers.image_tile_summary('inputs',
                tf.cast(inputs, dtype=tf.float32),
                rows=5,
                cols=5)

            recon = helpers.unflatten_tensor(
                model.reconstruct_images(flat_inputs), 
                shape)
            helpers.image_tile_summary('reconstructions',
                tf.cast(recon, dtype=tf.float32),
                rows=5,
                cols=5)

            samples = helpers.unflatten_tensor(
                model.generate_sample_images(config.num_samples), 
                shape)
            helpers.image_tile_summary('samples',
                tf.cast(samples, dtype=tf.float32),
                rows=5,
                cols=5)

    return model, loss


def run_train(config):
    """Runs the training of a latent variable model.

    Args:
        config: A configuration object with config values accessible as properties.
    """


    def create_graph():
        """Creates the training graph and loss to be optimised.

        Returns:
            train_loss: A float Tensor containing the training set's loss value.
            test_loss: A float Tensor containing the testing set's loss value.
            train_op: The training operation of the graph.
            global_step: The global step of the training process.
        """        

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf.compat.v1.logging.info("Loading the MNIST dataset...")

        train_images, train_labels, train_init = create_dataset(config, split='train', shuffle=True, repeat=True)
        test_images, test_labels, test_init = create_dataset(config, split='test', shuffle=False, repeat=True)

        # Get image shape for flattening on input to network
        img_shape = train_images.get_shape().as_list()[1:]

        tf.compat.v1.logging.info("Building the computation graph...")

        train_model, train_loss = create_model(config, split='train', inputs=train_images, labels=train_labels, shape=img_shape)
        test_model, test_loss = create_model(config, split='test', inputs=test_images, labels=test_labels, shape=img_shape)

        opt = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
        grads = opt.compute_gradients(train_loss, var_list=tf.trainable_variables())
        train_op = opt.apply_gradients(grads, global_step=global_step)

        tf.compat.v1.logging.info("Successfully built the graph!")

        return train_loss, train_init, test_loss, test_init, train_op, global_step


    with tf.Graph().as_default():
        if config.random_seed: 
            tf.set_random_seed(config.random_seed)

        with tf.device('/gpu:{}'.format(config.gpu_id)):
            train_loss, train_init, test_loss, test_init, train_op, global_step = create_graph()

            # Create session hooks
            log_hook = create_logging_hook(global_step, train_loss, test_loss, config.summarise_every)
            early_hook = helpers.EarlyStoppingHook(loss_op=test_loss,
                                   max_steps=config.early_stop_rounds,
                                   threshold=config.early_stop_threshold)

            # Set up the configuration for training
            config_proto = tf.ConfigProto(inter_op_parallelism_threads=1,
                                          intra_op_parallelism_threads=1)
            config_proto.gpu_options.allow_growth = True
            config_proto.gpu_options.per_process_gpu_memory_fraction = 0.5
            config_proto.log_device_placement = False
            config_proto.allow_soft_placement = True
            config_proto.gpu_options.visible_device_list = config.gpu_num

            # Set up log directory for saving checkpoints
            logdir = '{}/{}/h{}_n{}_z{}'.format(
                config.logdir, 
                config.model, 
                config.hidden_size,
                config.num_layers, 
                config.latent_size)
            if  not tf.io.gfile.exists(logdir):
                tf.compat.v1.logging.info("Creating log directory at {}".format(logdir))
                tf.io.gfile.makedirs(logdir)

            scaffold = tf.train.Scaffold(
                local_init_op=tf.group(
                    tf.local_variables_initializer(),
                    train_init,
                    test_init)
                )
            with tf.compat.v1.train.MonitoredTrainingSession(
                config=config_proto,
                scaffold=scaffold,
                hooks=[log_hook, early_hook],
                checkpoint_dir=logdir,
                save_checkpoint_secs=120,
                save_summaries_steps=config.summarise_every,
                log_step_count_steps=config.summarise_every) as sess:
                cur_step = -1

                while not sess.should_stop() and cur_step <= config.max_steps:
                    # Training
                    _, cur_step = sess.run([train_op, global_step])


def run_eval(config):
    """Runs the evaluation of a latent variable model.

    This method runs only one evaluation over the dataset, writes summaries to
    disk, and then terminates. It does not loop indefinitely.

    Args:
        config: A configuration object with config values accessible as properties.
    """

    def create_graph():
        """Creates the evaluation graph.

        Returns:
            sum_loss: A tuple of float Tensors containing the loss value
                summed across the entire batch.
            batch_size: An integer Tensor containing the batch size.
            z: The mean code resulting from transforming input images.
            samples: Samples from the model prior.
            sample_images: Sampled images from the model prior.
            labels: The labels associated with a batch of data.
            global_step: The global step the checkpoint was loaded from.
        """

        global_step = tf.compat.v1.train.get_or_create_global_step()

        tf.compat.v1.logging.info("Loading the MNIST dataset...")

        images, targets, img_shape, labels = create_dataset(config, split=config.split, shuffle=False, repeat=False)

        if config.model == 'gmvae':
            # Create a gmvae.TrainableGMVAE model object
            model = gmvae.create_gmvae(images.get_shape().as_list()[1],
                                       config.latent_size,
                                       mixture_components=config.mixture_components,
                                       fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                                       sigma_min=0.0,
                                       raw_sigma_bias=0.5,
                                       temperature=0.5)
        elif config.model == 'vae_gmp':
            # Create a mixture prior vae.TrainableVAE model object
            model = vae.create_vae(images.get_shape().as_list()[1],
                                   config.latent_size,
                                   mixture_components=config.mixture_components,
                                   fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                                   sigma_min=0.0,
                                   raw_sigma_bias=0.5)
        else:
            # Create a standard vae.TrainableVAE model object
            model = vae.create_vae(images.get_shape().as_list()[1],
                                   config.latent_size,
                                   fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                                   sigma_min=0.0,
                                   raw_sigma_bias=0.5)


        # Compute lower bounds on the log likelihood
        loss = model.run_model(images, targets)
        sum_loss = tf.reduce_sum(loss)
        # In case batches aren't divided evenly across dataset
        batch_size = tf.shape(images)[0]

        z = model.transform(images)
        samples = model.generate_samples(config.num_samples)
        sample_images = helpers.unflatten_tensor(
            model.generate_sample_images(config.num_samples, samples), 
            img_shape)

        if config.model == 'gmvae' or config.model == 'gmvae_alt':
            samples_y = model.generate_samples(config.num_samples, prior='y')
            sample_images_y = helpers.unflatten_tensor(
                model.generate_sample_images(config.num_samples, samples_y), 
                img_shape)
            sample_images = tf.stack((sample_images, sample_images_y), axis=0)

        return (sum_loss, batch_size, z, samples, sample_images, labels, global_step)


    def process_over_dataset(loss, batch_size, z, y, sess):
        """Process the dataset, averaging over the loss.

        Args:
            loss: Float Tensor containing the loss value evaluated on a single batch.
            batch_size: Integer Tensor containing the batch size. This can vary if the
                requested batch_size does not evenly divide the size of the dataset.
            z: The latent variables to accumulate over the dataset.
            y: The labels to accumulate over the dataset.
            sess: A TensorFlow Session object.
        Returns:
            avg_loss: A float containing the average loss value, normalised by 
                the number of examples in the dataset.
            latent_state: An np.array of the entire latent space.
            labels: An np.array of the dataset's labels.
        """

        total_loss = 0.0
        total_n_elems = 0.0
        latent_state = []
        labels = []

        while True:
            try:
                outs = sess.run([loss, batch_size, z, y])
            except tf.errors.OutOfRangeError:
                break

            total_loss += outs[0]
            total_n_elems += outs[1]
            latent_state.extend(np.reshape(outs[2], (-1, config.latent_size)))
            labels.extend(np.reshape(outs[3], (-1, 1)))

        avg_loss = total_loss / total_n_elems

        return avg_loss, np.array(latent_state), np.array(labels)


    def summarise_loss(avg_loss, summary_writer, step):
        """Creates log-likelihood lower bound summaries and writes them to disk.

        Args:
            avg_loss: A python float, contains the value of the
                evaluated loss normalised by the number of examples.
            summary_writer: A tf.SummaryWriter.
            step: The current global step.
        """

        def scalar_summary(name, value):
            value = tf.Summary.Value(tag=name, simple_value=value)
            return tf.Summary(value=[value])


        per_example_summary = scalar_summary("%s/loss_per_example" % config.split, avg_loss)
        summary_writer.add_summary(per_example_summary, global_step=step)
        summary_writer.flush()


    def plot_latent(path, step, latent_var, labels, ndim=2):
        feat_cols = ['index' + str(i) for i in range(latent_var.shape[1])]
        df = pd.DataFrame(latent_var, columns=feat_cols)
        df['y'] = labels
        df['z1-tsne'] = latent_var[:,0]
        df['z2-tsne'] = latent_var[:,1]

        if ndim == 2:
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x='z1-tsne', y='z2-tsne',
                hue='y',
                palette=sns.color_palette('hls', config.mixture_components),
                data=df,
                legend='full',
                alpha=0.3)
        elif ndim == 3:
            df['z3-tsne'] = latent_var[:,2]

            ax = plt.figure(figsize=(16,10)).gca(projection='3d')
            ax.scatter(
                xs=df['z1-tsne'], 
                ys=df['z2-tsne'], 
                zs=df['z3-tsne'], 
                c=df['y'], 
                cmap='tab10'
            )
            ax.set_xlabel("z1-tsne")
            ax.set_ylabel("z2-tsne")
            ax.set_zlabel("z3-tsne")
        else:
            tf.compat.v1.logging.error("Cannot accommodate that many dimensions!")

        plt.savefig('{}/step_{}_dim_{}'.format(path, step, ndim))
        plt.show()


    def plot_prior_samples(path, step, samples):
        # Prior samples plot
        g = sns.jointplot(
            x=samples[..., 0],
            y=samples[..., 1],
            kind='scatter',
            stat_func=None,
            marginal_kws=dict(bins=50))
        g.set_axis_labels("$x_1$", "$x_2$");

        plt.savefig('{}/step_{}_samples'.format(path, step))
        plt.show()


    def display_images(path, step, images, name='sample', n=10):
        plt.ioff()
        fig, axs = plt.subplots(n, n, figsize=(10, 10))

        for i in range(n):
            for j in range(n):
                axs[i, j].imshow(images[i*n+j].squeeze(), interpolation='none', cmap='gray')
                axs[i, j].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0)

        plt.savefig('{}/step_{}_{}_images'.format(path, step, name))
        plt.show()
        plt.ion()


    with tf.Graph().as_default():
        if config.random_seed: 
            tf.set_random_seed(config.random_seed)

        loss, bs, z, samples, images, y, global_step = create_graph()

        # Set up log directory for loading checkpoints
        logdir = '{}/{}/h{}_n{}_z{}'.format(
            config.logdir, 
            config.model, 
            config.hidden_size,
            config.num_layers, 
            config.latent_size)

        # Set up the summary directory for storing results on split
        summary_dir = '{}/{}'.format(
            logdir,
            config.split)
        summary_writer = tf.compat.v1.summary.FileWriter(summary_dir, flush_secs=15, max_queue=100)

        saver = tf.compat.v1.train.Saver()
        with tf.compat.v1.train.SingularMonitoredSession() as sess:
            helpers.wait_for_checkpoint(saver, sess, logdir)
            step = sess.run(global_step)
            tf.compat.v1.logging.info("Model restored from step %d" % step)

            avg_loss, z_out, y_out = process_over_dataset(loss, bs, z, y, sess)
            summarise_loss(avg_loss, summary_writer, step)

            tf.compat.v1.logging.info("%s loss/example: %f", config.split, avg_loss)

            tf.compat.v1.logging.info("Plotting latent code!")
            z_two = reduce_dimensionality(z_out[:config.num_samples])
            plot_latent(summary_dir, step, z_two, y_out[:config.num_samples])
            #z_three = reduce_dimensionality(z_out[:config.num_samples], dim=3)
            #plot_latent(summary_dir, step, z_three, y_out[:config.num_samples], ndim=3)

            samples_out, images_out = sess.run([samples, images])
            samples_two = reduce_dimensionality(samples_out)

            tf.compat.v1.logging.info("Plotting prior samples!")
            plot_prior_samples(summary_dir, step, samples_two)

            if config.model == 'gmvae' or config.model == 'gmvae_alt':
                display_images(summary_dir, step, images_out[0])
                display_images(summary_dir, step, images_out[1], name='sample_y')
            else:
                display_images(summary_dir, step, images_out)