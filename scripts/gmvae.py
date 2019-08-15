from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

import base


class GMVAE(object):
    """Implementation of a Gaussian Mixture Variational Autoencoder (GMVAE)."""

    def __init__(self,
                 prior_y,
                 prior_gmm,
                 decoder,
                 encoder_y,
                 encoder_mvn):
        """Create a GMVAE.

        Args:
            prior_y: A tf.distributions.Distribution that implements p(y).
            prior_gmm: A callable that implements the prior distribution p(z | y)
                Must accept as argument the y discrete variable and return
                a tf.distributions.MixtureSameFamily distribution.
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the encoded latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder_y: A callable that implements the inference q(y | x).
            encoder_mvn: A callable that implements the inference q(z | x, y).
        """

        self._prior_y = prior_y
        self._prior_gmm = prior_gmm
        self._decoder = decoder
        self._encoder_y = encoder_y
        self._encoder_mvn = encoder_mvn


    def prior_y(self):
        """Getter for the prior distribution p(y).

        Returns:
            p(y): A distribution with shape [batch_size, mixture_components].
        """

        return self._prior_y


    def prior_gmm(self, y, logits):
        """Computes the GMM prior distribution p(z | y).

        Args:
            y: The discrete intermediate variable y.
            logits: Selection distribution for mixture.

        Returns:
            p(z | y): A GMM distribution with shape [batch_size, latent_size].
        """

        return self._prior_gmm(y, logits=logits)


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
            q(y | x): A distribution with shape [batch_size, mix_components].
        """

        x = tf.cast(x, dtype=tf.float32)

        return self._encoder_y(x)


    def encoder_mvn(self, x, y):
        """Computes the inference distribution q(z | x, y).


        Args:
            x: Input images of shape [batch_size, data_size].
            y: Discrete variable y of shape [batch_size, mix_components].

        Returns:
            q(z | x, y): A distribution with shape [batch_size, latent_size].
        """
        
        x = tf.cast(x, dtype=tf.float32)

        return self._encoder_mvn(x, y)


class TrainableGMVAE(GMVAE):
    """A GMVAE subclass with methods for training and evaluation."""

    def __init__(self,
                 mix_components,
                 prior_y,
                 prior_gmm,
                 decoder,
                 encoder_y,
                 encoder_mvn,
                 random_seed=None):
        """Create a trainable GMVAE.

        Args:
            mix_components: The number of mixture components.
            prior_y: A tf.distributions.Distribution that implements p(y).
            prior_gmm: A callable that implements the prior distribution p(z | y)
                Must accept as argument the y discrete variable and return
                a tf.distributions.MixtureSameFamily distribution.
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the encoded latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder_y: A callable that implements the inference q(y | x).
            encoder_mvn: A callable that implements the inference q(z | x, y).
            random_seed: The seed for the random ops.
        """

        super(TrainableGMVAE, self).__init__(
            prior_y, prior_gmm, decoder,
            encoder_y, encoder_mvn)
        self.mix_components = mix_components
        self.random_seed = random_seed


    def reconstruct_images(self, images):
        """Generate reconstructions of 'images' from the model."""

        q_y = self.encoder_y(images)
        y = tf.cast(q_y.sample(seed=self.random_seed), dtype=tf.float32)

        q_z = self.encoder_mvn(images, y)
        z = q_z.sample(seed=self.random_seed)

        p_x_given_z = self.decoder(z)
        recon = p_x_given_z.mean(name='reconstructions')

        return recon


    def generate_sample_images(self, num_samples, z=None):
        """Generates mean sample images from the latent space.

        Can provide latent variable 'z' to generate data for
        this point in the latent space. Else draw from prior.
        """

        if z is None:
            z = self.generate_samples(num_samples)

        p_x_given_z = self.decoder(z)
        sample_images = p_x_given_z.mean(name='sample_images')

        return sample_images


    def transform(self, inputs):
        """Transform 'inputs' to yield mean latent code."""

        q_y = self.encoder_y(inputs)
        y = q_y.mean()

        q_z = self.encoder_mvn(inputs, y)
        z = q_z.mean(name='code')

        return z


    def generate_samples(self, num_samples, prior='gmm'):
        """Generate 'num_samples' samples from the model's 'prior'."""

        p_y = self.prior_y()

        if prior == 'gmm':
            # Setting 'y' to zero
            zero_input = tf.expand_dims(
                tf.zeros([self.mix_components]),
                axis=0, 
                name='zero_y')
            p_z_given_y = self.prior_gmm(zero_input, p_y.logits)
            # Sampling from GMM rather than p(y)
            z = p_z_given_y.sample(num_samples, seed=self.random_seed, name='samples')
            # Squeeze out artificial extra dimension
            z = tf.squeeze(z)
        elif prior == 'y':
            y = tf.cast(p_y.sample(num_samples, seed=self.random_seed), dtype=tf.float32, name='samples_y')
            p_z_given_y = self.prior_gmm(y, p_y.logits)        
            z = p_z_given_y.sample(seed=self.random_seed)
        else:
            tf.logging.error("No prior by that name: %s", prior)

        return z


    def run_model(self, images, targets):
        """Runs the model and computes weights for a batch of images and targets.

        Args:
            images: A batch of images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].
            targets: A batch of target images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].

        Returns:
            loss: A float loss Tensor.
        """

        # Prior p(y) with uniform distribution 
        p_y = self.prior_y()
        # Encoder accepts images x and implements q(y | x)
        q_y = self.encoder_y(images)
        y = tf.cast(q_y.sample(seed=self.random_seed), dtype=tf.float32)

        # Prior accepts y as input and implements p(z | y)
        # One-hot vector sampled from uniform distribution, p(y) logits
        p_z_given_y = self.prior_gmm(y, p_y.logits)
        # Encoder accept images x and y as inputs to implement q(z | x, y)
        q_z = self.encoder_mvn(images, y)
        z = q_z.sample(seed=self.random_seed)

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        # Reconstruction loss term
        log_p_x_given_z = p_x_given_z.log_prob(targets)
        tf.summary.scalar('nll_scalar', tf.reduce_mean(log_p_x_given_z))

        # Latent loss between approximate posterior and prior for z
        log_p_z_given_y = p_z_given_y.log_prob(z)
        log_q_z = q_z.log_prob(z)
        tf.summary.scalar('latent_loss', tf.reduce_mean(log_p_z_given_y - log_q_z))
        
        # Conditional entropy loss
        log_q_y = q_y.log_prob(y)     
        tf.summary.scalar('cond_entropy', -tf.reduce_mean(log_q_y))

        # Need to maximise the ELBO with respect to these weights:
        loss = -tf.reduce_mean(log_p_x_given_z + log_p_z_given_y - log_q_z - log_q_y)
        tf.summary.scalar('elbo', -loss)

        return loss


def create_gmvae(
    data_size,
    latent_size,
    mixture_components=1,
    fcnet_hidden_sizes=None,
    hidden_activation_fn=tf.nn.relu,
    sigma_min=0.001,
    raw_sigma_bias=0.25,
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
        random_seed: A random seed for the VRNN resampling operations.

    Returns:
        model: A TrainableGMVAE object.
    """

    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]

    # Prior p(y) is a uniform distribution
    prior_y = tfp.distributions.OneHotCategorical(
        logits=tf.zeros([mixture_components]),
        name='prior_y')
    # Prior p(z | y) is a learned mixture of Gaussians, where mu and
    # sigma are output from a fully connected network conditioned on y
    prior_gmm = base.GaussianMixture(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        mixture_components=mixture_components,
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
    encoder_y = base.ConditionalCategorical(
        size=mixture_components,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        name='encoder_y')
    # A callable that implements the inference distribution q(z | x, y)
    encoder_mvn = base.ConditionalNormal(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        name='encoder_mvn')

    return TrainableGMVAE(mixture_components, prior_y, prior_gmm,
                          decoder, encoder_y, encoder_mvn,
                          random_seed=random_seed)