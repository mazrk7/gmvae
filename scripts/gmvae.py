from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import base
import utils


class GMVAE(object):
    """Implementation of a Gaussian Mixture Variational Autoencoder (GMVAE)."""

    def __init__(self,
                 prior_gmm,
                 decoder,
                 encoder_y,
                 encoder_gmm):
        """Create a GMVAE.

        Args:
            prior_gmm: A callable that implements the prior distribution p(z | y)
                Must accept as argument the y discrete variable and return
                a tf.distributions.MultivariateNormalDiag distribution.
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the encoded latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder_y: A callable that implements the inference q(y | x) over
                the discrete latent variable y.
            encoder_gmm: A callable that implements the inference q(z | x, y) over
                the continuous latent variable z.
        """

        self._prior_gmm = prior_gmm
        self._decoder = decoder
        self._encoder_y = encoder_y
        self._encoder_gmm = encoder_gmm


    def prior_gmm(self, y):
        """Computes the GMM prior distribution p(z | y).

        Args:
            y: The discrete intermediate variable y.

        Returns:
            p(z | y): A GMM distribution with shape [batch_size, latent_size].
        """

        return self._prior_gmm(y)


    def decoder(self, z):
        """Computes the generative distribution p(x | z).

        Args:
            z: The stochastic latent state z.

        Returns:
            p(x | z): A distribution with shape [batch_size, data_size].
        """

        return self._decoder(z)


    def encoder_y(self, x):
        """Computes the inference distribution q(y | x).

        Args:
            x: The input images to the inference network.

        Returns:
            q(y | x): A RelaxedOneHotCategorical distribution with shape 
                [batch_size, mix_components].
        """

        x = tf.cast(x, dtype=tf.float32)

        return self._encoder_y(x)


    def encoder_gmm(self, x, y):
        """Computes the inference distribution q(z | x, y).


        Args:
            x: Input images of shape [batch_size, data_size].
            y: Discrete variable y of shape [batch_size, mix_components].

        Returns:
            q(z | x, y): A MultivariateNormalDiag distribution with shape 
                [batch_size, latent_size].
        """
        
        x = tf.cast(x, dtype=tf.float32)

        return self._encoder_gmm(x, y)


class TrainableGMVAE(GMVAE):
    """A GMVAE subclass with methods for training and evaluation."""

    def __init__(self,
                 mix_components,
                 prior_gmm,
                 decoder,
                 encoder_y,
                 encoder_gmm,
                 random_seed=None):
        """Create a trainable GMVAE.

        Args:
            mix_components: The number of mixture components.
            prior_gmm: A callable that implements the prior distribution p(z | y)
                Must accept as argument the y discrete variable and return
                a tf.distributions.MultivariateNormalDiag distribution.
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the encoded latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder_y: A callable that implements the inference q(y | x).
            encoder_gmm: A callable that implements the inference q(z | x, y).
            random_seed: The seed for the random ops.
        """

        super(TrainableGMVAE, self).__init__(
            prior_gmm, decoder,
            encoder_y, encoder_gmm)

        self.mix_components = mix_components
        self.random_seed = random_seed


    def reconstruct_images(self, images):
        """Generate reconstructions of 'images' from the model."""

        q_y = self.encoder_y(images)
        y = q_y.sample(seed=self.random_seed)

        q_z = self.encoder_gmm(images, y)
        z = q_z.sample(seed=self.random_seed)

        p_x_given_z = self.decoder(z)
        recon = p_x_given_z.mean(name='reconstructions')

        return recon


    def generate_sample_images(self, z=None, batch_size=1, num_samples=1, prior='gmm'):
        """Generates mean sample images from the model.

        Can provide latent variable 'z' to generate data for
        this point in the latent space. Else draw from prior.
        """

        if z is None:
            z = self.generate_samples(batch_size, num_samples, prior)

        p_x_given_z = self.decoder(z)
        sample_images = p_x_given_z.mean(name=prior + '_sample_images')

        return sample_images


    def transform(self, inputs):
        """Transform 'inputs' to yield mean latent code."""

        q_y = self.encoder_y(inputs)
        y = q_y.sample(seed=self.random_seed)

        q_z = self.encoder_gmm(inputs, y)
        z = q_z.mean(name='code')

        return z


    def generate_samples(self, batch_size, num_samples, prior='gmm'):
        """Generate 'num_samples' samples from the model's 'prior'."""

        # Initialise discrete variable y to index clusters
        y_ = tf.fill(tf.stack([batch_size, self.mix_components]), 0.0)
        if prior == 'gmm':
            # Generate random outputs for each category
            z_k = [None] * self.mix_components
            for k in xrange(self.mix_components):
                y = tf.add(y_, tf.eye(self.mix_components, dtype=tf.float32)[k])

                p_z_given_y = self.prior_gmm(y)
                samples = p_z_given_y.sample(num_samples, seed=self.random_seed)
                
                # Average along batch dimension
                z_k[k] = tf.reduce_mean(samples, axis=1)

            # Concatenate outputs into Tensor of shape [samples, latent_size]
            z = tf.concat(z_k, axis=0)
        elif prior == 'y':
            # Draw from a random cluster
            k = np.random.randint(0, high=self.mix_components)

            # Generate random outputs from one specific category
            y = tf.add(y_, tf.eye(self.mix_components, dtype=tf.float32)[k])

            p_z_given_y = self.prior_gmm(y)
            samples = p_z_given_y.sample(num_samples*self.mix_components, seed=self.random_seed)

            # Average along batch dimension
            z = tf.reduce_mean(samples, axis=1)
        else:
            tf.compat.v1.logging.error("No prior by that name: %s", prior)

        return z


    def run_model(self, images, targets, labels):
        """Runs the model and computes weights for a batch of images and targets.

        Args:
            images: A batch of images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].
            targets: A batch of target images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].
            labels: A batch of int labels to evaluate clustering performance.

        Returns:
            loss: A float loss Tensor.
        """

        # Encoder accepts images x and implements q(y | x)
        q_y = self.encoder_y(images)
        # Sample categorical variable y from the Gumbel-Softmax distribution
        y = q_y.sample(seed=self.random_seed)

        # Prior accepts y as input and implements p(z | y)
        p_z_given_y = self.prior_gmm(y)

        # Encoder accept images x and y as inputs to implement q(z | x, y)
        q_z = self.encoder_gmm(images, y)
        # Sample latent Gaussian variable z
        z = q_z.sample(seed=self.random_seed)

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        # Reconstruction loss term i.e. the negative log-likelihood
        nll = -tf.reduce_mean(p_x_given_z.log_prob(targets))
        tf.compat.v1.summary.scalar('nll_scalar', nll)

        # Latent loss between approximate posterior and prior for z
        kl_div_z = tf.reduce_mean(q_z.log_prob(z) - p_z_given_y.log_prob(z))
        tf.compat.v1.summary.scalar('kl_div_z', kl_div_z)
        
        # Conditional entropy loss
        nent = -tf.reduce_mean(utils.entropy(
            q_y.distribution.logits, tf.nn.softmax(q_y.distribution.logits)))
        tf.compat.v1.summary.scalar('nent', nent)

        # Need to maximise the ELBO with respect to these weights
        loss = nll + kl_div_z + nent
        tf.compat.v1.summary.scalar('elbo', -loss)

        # Keep track of the clustering accuracy during training
        cluster_acc = utils.cluster_acc(q_y.distribution.logits, labels, self.mix_components)
        tf.compat.v1.summary.scalar('cluster_acc', cluster_acc)

        return loss


def create_gmvae(
    data_size,
    latent_size,
    mixture_components=1,
    fcnet_hidden_sizes=None,
    hidden_activation_fn=tf.nn.relu,
    sigma_min=0.001,
    raw_sigma_bias=0.25,
    temperature=1.0,
    random_seed=None):
    """A factory method for creating GMVAEs.

    Args:
        data_size: The dimension of the vector that makes up the flattened images.
        latent_size: The size of the stochastic latent state of the GMVAE.
        mixture_components: The number of mixture components.
        fcnet_hidden_sizes: A list of python integers, the size of the hidden
            layers of the fully connected networks that parameterize the conditional
            distributions of the GMVAE. If None, then it defaults to one hidden
            layer of size latent_size.
        hidden_activation_fn: The activation operation applied to intermediate 
            layers, and optionally to the output of the final layer.
        sigma_min: The minimum value that the standard deviation of the
            distribution over the latent state can take.
        raw_sigma_bias: A scalar that is added to the raw standard deviation
            output from the neural networks that parameterize the prior and
            approximate posterior. Useful for preventing standard deviations close to zero.
        temperature: Degree of how approximately discrete the Gumbel distribution that models
            the discrete latent variable y should be.
        random_seed: A random seed for the VRNN resampling operations.

    Returns:
        model: A TrainableGMVAE object.
    """

    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]

    # Prior p(z | y) is a learned mixture of Gaussians, where mu and
    # sigma are output from a fully connected network conditioned on y
    prior_gmm = base.ConditionalNormal(
        size=latent_size,
        hidden_layer_sizes=None,
        hidden_activation_fn=hidden_activation_fn,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        name='prior_gmm')

    # The generative distribution p(x | z) is conditioned on the latent
    # state variable z via a fully connected network
    decoder = base.ConditionalBernoulli(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        name='decoder')

    # A callable that implements the inference distribution q(y | x)
    # Use the Gumbel-Softmax distribution to model the categorical latent variable
    encoder_y = base.ConditionalCategorical(
        size=mixture_components,
        temperature=temperature,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        name='encoder_y')
    # A callable that implements the inference distribution q(z | x, y)
    encoder_gmm = base.ConditionalNormal(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        name='encoder_gmm')

    return TrainableGMVAE(mixture_components, prior_gmm,
        decoder, encoder_y, encoder_gmm, random_seed=random_seed)