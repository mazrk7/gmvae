from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

import utils
import gmvae
import vae


def create_dataset(config, split, handle, shuffle, repeat):
    """Creates the MNIST dataset for a given config.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object. This function expects the properties
            batch_size, dataset_path, and latent_size to be defined.
        split: The dataset split to load.
        handle: A tf.placeholder for the iterator handle.
        shuffle: If true, shuffle the dataset randomly.
        repeat: If true, repeat the dataset endlessly.

    Returns:
        images: A batch of image sequences represented as a dense Tensor 
            of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1].
        labels: A batch of integer labels, for use in evaluation of clustering.
        iter_str: A string handle to the split set's iterator.
    """

    dataset, datasets_info = tfds.load(name='mnist',
                                      split=split,
                                      with_info=True,
                                      as_supervised=False)


    def _preprocess(sample):
        image = tf.cast(sample['image'], tf.float32) / 255.  # Scale to unit interval
        image = image < tf.random.uniform(tf.shape(image))
        return image, sample['label']


    dataset = (dataset.map(_preprocess, num_parallel_calls=8)
                      .batch(config.batch_size)
                      .prefetch(tf.data.experimental.AUTOTUNE))
    
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(datasets_info.splits[split].num_examples)

    iterator = tf.compat.v1.data.Iterator.from_string_handle(
        handle, dataset.output_types, dataset.output_shapes)
    images, labels = iterator.get_next()
    
    iter_str = tf.compat.v1.data.make_one_shot_iterator(dataset).string_handle()

    return images, labels, iter_str


def create_model(config, data_dim):
    """Creates either a gmvae.TrainableGMVAE or vae.TrainableVAE model object.

    Args:
        config: A configuration object with config values accessible as properties.
            Most likely a FLAGS object. This function expects the properties
            batch_size, dataset_path, and latent_size to be defined.
        data_dim: An integer representing the input dimensionality to the network.

    Returns:
        model: The trainable model.
    """

    if config.model == 'gmvae':
        # Create a gmvae.TrainableGMVAE model object
        model = gmvae.create_gmvae(data_dim,
                                   config.latent_size,
                                   mixture_components=config.mixture_components,
                                   fcnet_hidden_sizes=[config.hidden_size] * config.num_layers,
                                   sigma_min=0.0,
                                   raw_sigma_bias=0.5,
                                   temperature=1.0)
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

    return model


def run_train(config):
    """Runs the training of a latent variable model.

    Args:
        config: A configuration object with config values accessible as properties.
    """

    def create_model_loss(split, model, inputs, labels, shape):
        """Creates a loss tensor associated with the provided model during training.

        Args:
            split: The dataset split to load.
            model: The model object to compute the loss over.
            inputs: A batch of input sequences represented as a dense Tensor 
                of shape [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1].
            labels: A batch of integer labels to evaluate clustering performance.
            shape: A list representing the shape of the input images.

        Returns:
            loss: A float Tensor representing the model loss.
        """

        flat_inputs = utils.flatten_tensor(inputs, shape, name=split + '_inputs')

        with tf.name_scope(split):
            if config.model == 'gmvae':
                loss = model.run_model(flat_inputs, flat_inputs, labels)
                sampled_z = model.generate_samples(batch_size=config.batch_size, num_samples=1)
            else:
                loss = model.run_model(flat_inputs, flat_inputs)
                sampled_z = model.generate_samples(num_samples=10)

            with tf.name_scope('image_summaries'):
                utils.image_tile_summary('inputs',
                    tf.cast(inputs, dtype=tf.float32),
                    rows=5,
                    cols=5)

                recon = utils.unflatten_tensor(
                    model.reconstruct_images(flat_inputs), 
                    shape)
                utils.image_tile_summary('reconstructions',
                    tf.cast(recon, dtype=tf.float32),
                    rows=5,
                    cols=5)

                samples = utils.unflatten_tensor(
                    model.generate_sample_images(z=sampled_z), 
                    shape)
                utils.image_tile_summary('samples',
                    tf.cast(samples, dtype=tf.float32),
                    rows=3,
                    cols=3)

        return loss


    def create_training_graph(handle):
        """Creates the training graph and loss to be optimised.

        Args:
            handle: A tf.placeholder for the iterator handle.

        Returns:
            loss: A float Tensor containing the training set's loss value.
            iterator: The training set's string iterator handle.
            model: The trainable model object.
            train_op: The training operation of the graph.
            global_step: The global step of the training process.
        """        

        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            tf.compat.v1.logging.info("Loading the MNIST training set...")
            images, labels, iterator = create_dataset(config, 
                split='train', handle=handle, shuffle=True, repeat=True)

        tf.compat.v1.logging.info("Building the training graph...")
        img_shape = images.get_shape().as_list()[1:]
        model = create_model(config, data_dim=np.prod(img_shape))
        loss = create_model_loss('train', model, inputs=images, labels=labels, shape=img_shape)

        opt = tf.compat.v1.train.AdamOptimizer(config.learning_rate)
        grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
        train_op = opt.apply_gradients(grads, global_step=global_step)
        tf.compat.v1.logging.info("Successfully built the training graph!")

        return loss, iterator, model, train_op, global_step


    def create_evaluation_graph(handle, model):
        """Creates the graph to evaluate the provided latent variable 'model'.

        Args:
            handle: A tf.placeholder for the iterator handle.
            model: A trainable VAE or GMVAE model.

        Returns:
            loss: A float Tensor containing the test set's loss value.
            iterator: The test set's string iterator handle.
        """        

        with tf.device('/cpu:0'):
            tf.compat.v1.logging.info("Loading the MNIST test set...")
            images, labels, iterator = create_dataset(config, 
                split='test', handle=handle, shuffle=False, repeat=True)

        tf.compat.v1.logging.info("Building the evaluation graph...")
        img_shape = images.get_shape().as_list()[1:]
        loss = create_model_loss('test', model, inputs=images, labels=labels, shape=img_shape)
        tf.compat.v1.logging.info("Successfully built the evaluation graph!")

        return loss, iterator


    def step_fn(step_context):
        # Train the model
        _, cur_step = step_context.session.run([train_op, global_step], feed_dict={handle: ds_handle_hook.train_handle})

        # Return evaluation of the model
        return cur_step, step_context.run_with_hooks(test_loss, feed_dict={handle: ds_handle_hook.valid_handle})


    with tf.Graph().as_default():
        if config.random_seed: 
            tf.set_random_seed(config.random_seed)
        
        with tf.device('/gpu:{}'.format(config.gpu_id)):
            # Placeholder for the dataset iterator handle
            handle = tf.placeholder(tf.string, shape=[])

            train_loss, train_iter, model, train_op, global_step = create_training_graph(handle)
            test_loss, test_iter = create_evaluation_graph(handle, model)

            # Create session hooks
            ds_handle_hook = utils.DatasetHandleHook(train_iter, test_iter)
            log_hook = utils.create_logging_hook(global_step, train_loss, test_loss, config.summarise_every)
            early_hook = utils.EarlyStoppingHook(loss_op=test_loss,
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

            with tf.compat.v1.train.MonitoredTrainingSession(
                config=config_proto,
                hooks=[ds_handle_hook, log_hook, early_hook],
                checkpoint_dir=logdir,
                save_checkpoint_secs=120,
                save_summaries_steps=config.summarise_every,
                log_step_count_steps=config.summarise_every) as sess:
                cur_step = -1

                while not sess.should_stop() and cur_step <= config.max_steps:
                    cur_step, _ = sess.run_step_fn(step_fn)


def run_eval(config):
    """Runs the evaluation of a latent variable model.

    This method runs only one evaluation over the dataset, writes summaries to
    disk, and then terminates. It does not loop indefinitely.

    Args:
        config: A configuration object with config values accessible as properties.
    """

    def create_graph(handle):
        """Creates the evaluation graph.

        Args:
            handle: A tf.placeholder for the iterator handle.

        Returns:
            sum_loss: A tuple of float Tensors containing the loss value
                summed across the entire batch.
            batch_size: An integer Tensor containing the batch size.
            z: The mean code resulting from transforming input images.
            samples: Samples from the model prior.
            sample_images: Sampled images from the model prior.
            labels: The labels associated with a batch of data.
            iterator: The split set's string iterator handle.
            global_step: The global step the checkpoint was loaded from.
        """

        global_step = tf.compat.v1.train.get_or_create_global_step()

        with tf.device('/cpu:0'):
            tf.compat.v1.logging.info("Loading the MNIST dataset...")
            images, labels, iterator = create_dataset(config, 
                split=config.split, handle=handle, shuffle=False, repeat=False)

        # Get image shape for flattening on input to network
        img_shape = images.get_shape().as_list()[1:]
        model = create_model(config, data_dim=np.prod(img_shape))
        
        flat_inputs = utils.flatten_tensor(images, img_shape)

        # Batch size in case dataset isn't exactly divisible
        batch_size = tf.shape(flat_inputs)[0]

        # Transform inputs into latent code z
        z = model.transform(flat_inputs)

        if config.model == 'gmvae':
            # Compute lower bounds on the log likelihood
            loss = model.run_model(flat_inputs, flat_inputs, labels)

            # Generate random samples from the GMM prior
            samples = model.generate_samples(batch_size=config.batch_size, 
                num_samples=config.num_samples)
            sample_images_gmm = utils.unflatten_tensor(
                model.generate_sample_images(batch_size=config.batch_size, 
                    num_samples=config.num_generations), 
                img_shape)
            # Generate random samples from the prior given a single category
            sample_images_y = utils.unflatten_tensor(
                model.generate_sample_images(batch_size=config.batch_size, 
                    num_samples=config.num_generations, prior='y'), 
                img_shape)
            sample_images = tf.stack((sample_images_gmm, sample_images_y), axis=0)
        else:
            # Compute lower bounds on the log likelihood
            loss = model.run_model(flat_inputs, flat_inputs)

            # Generate samples from the model prior
            samples = model.generate_samples(num_samples=config.num_samples)
            sample_images = utils.unflatten_tensor(
                model.generate_sample_images(num_samples=config.num_generations), 
                img_shape)

        return (tf.reduce_sum(loss), batch_size, z, samples, sample_images, labels, iterator, global_step)


    def process_over_dataset(loss, batch_size, z, y, sess, iter_str):
        """Process the dataset, averaging over the loss.

        Args:
            loss: Float Tensor containing the loss value evaluated on a single batch.
            batch_size: Integer Tensor containing the batch size. This can vary if the
                requested batch_size does not evenly divide the size of the dataset.
            z: The latent variables to accumulate over the dataset.
            y: The labels to accumulate over the dataset.
            sess: A TensorFlow Session object.
            iter_str: The split set's string iterator handle.
            
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
                outs = sess.run([loss, batch_size, z, y], feed_dict={handle: iter_str})
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


    def plot_latent(path, step, latent_var, labels):
        # Project inputs into latent space
        feat_cols = ['index' + str(i) for i in range(latent_var.shape[1])]
        df = pd.DataFrame(latent_var, columns=feat_cols)
        df['y'] = labels
        df['z1-tsne'] = latent_var[:,0]
        df['z2-tsne'] = latent_var[:,1]

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x='z1-tsne', y='z2-tsne',
            hue='y',
            palette=sns.color_palette('hls', config.mixture_components),
            data=df,
            legend='full',
            alpha=0.3)

        plt.savefig('{}/step_{}'.format(path, step))
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

        # Placeholder for the dataset iterator handle
        handle = tf.placeholder(tf.string, shape=[])

        loss, bs, z, samples, images, y, iter_str, global_step = create_graph(handle)

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
            utils.wait_for_checkpoint(saver, sess, logdir)
            step, iter_str_out = sess.run([global_step, iter_str])

            tf.compat.v1.logging.info("Model restored from step %d" % step)

            avg_loss, z_out, y_out = process_over_dataset(loss, bs, z, y, sess, iter_str_out)
            summarise_loss(avg_loss, summary_writer, step)

            tf.compat.v1.logging.info("%s loss/example: %f", config.split, avg_loss)

            tf.compat.v1.logging.info("Plotting latent code!")
            z_two = utils.reduce_dimensionality(z_out)
            plot_latent(summary_dir, step, z_two, y_out)

            samples_out, images_out = sess.run([samples, images], feed_dict={handle: iter_str_out})
            samples_two = utils.reduce_dimensionality(samples_out)

            tf.compat.v1.logging.info("Plotting prior samples!")
            plot_prior_samples(summary_dir, step, samples_two)

            if config.model == 'gmvae':
                display_images(summary_dir, step, images_out[0])
                display_images(summary_dir, step, images_out[1], name='sample_y')
            else:
                display_images(summary_dir, step, images_out)