from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

import base


class VAE(object):
    """Implementation of a Variational Autoencoder (VAE)."""

    def __init__(self,
                 prior,
                 decoder,
                 encoder):
        """Create a VAE.

        Args:
            prior: A tf.distributions.Distribution that implements p(z).
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder: A callable that implements the inference q(z | x).
        """

        self._prior = prior
        self._decoder = decoder
        self._encoder = encoder


    def prior(self):
        """Getter for the prior distribution p(z).

        Returns:
            p(z): A distribution with shape [batch_size, latent_size].
        """

        return self._prior


    def decoder(self, z):
        """Computes the generative distribution p(x | z).

        Args:
            z: The stochastic latent state z.

        Returns:
            p(x | z): A distribution with shape [batch_size, data_size].
        """

        return self._decoder(z)


    def encoder(self, x):
        """Computes the inference distribution q(z | x).


        Args:
            x: Images of shape [batch_size, data_size].

        Returns:
            q(z | x): A distribution with shape [batch_size, latent_size].
        """
        
        x = tf.cast(x, dtype=tf.float32)

        return self._encoder(x)


class TrainableVAE(VAE):
    """A VAE subclass with methods for training and evaluation."""

    def __init__(self,
                 prior,
                 decoder,
                 encoder,
                 mix_components=1,
                 random_seed=None):
        """Create a trainable VAE.

        Args:
            prior: A tf.distributions.Distribution that implements p(z).
            decoder: A callable that implements the generative distribution
                p(x | z). Must accept as arguments the latent state z
                and return a subclass of tf.distributions.Distribution that 
                can be used to evaluate the log_prob of the targets.
            encoder: A callable that implements the inference q(z | x).
            mix_components: The number of components in the mixture prior.
            random_seed: The seed for the random ops.
        """

        super(TrainableVAE, self).__init__(prior, decoder, encoder)
        self.mix_components = mix_components
        self.random_seed = random_seed


    def reconstruct_images(self, images):
        """Generate reconstructions of 'images' from the model."""

        q_z = self.encoder(images)
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

        q_z = self.encoder(inputs)
        z = q_z.mean(name='code')

        return z


    def generate_samples(self, num_samples):
        """Generate 'num_samples' samples from the model prior."""

        p_z = self.prior()
        z = p_z.sample(num_samples, seed=self.random_seed, name='samples')

        return z


    def run_model(self, images, targets, batch_size):
        """Runs the model and computes weights for a batch of images and targets.

        Args:
            images: A batch of images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].
            targets: A batch of target images generated from a dataset iterator
                and with Tensor shape [batch_size, data_size].
            batch_size: Batch size. Unused here.

        Returns:
            loss: A float loss Tensor.
        """

        # Prior with Gaussian distribution p(z)
        p_z = self.prior()

        # Encoder accept images x implement q(z | x)
        q_z = self.encoder(images)
        z = q_z.sample(seed=self.random_seed)

        # Generative distribution p(x | z)
        p_x_given_z = self.decoder(z)

        log_p_x_given_z = p_x_given_z.log_prob(targets)
        tf.summary.scalar('nll_scalar', tf.reduce_mean(log_p_x_given_z))

        log_p_z = p_z.log_prob(z)
        log_q_z = q_z.log_prob(z)
        tf.summary.scalar('kl_div_scalar', tf.reduce_mean(log_p_z - log_q_z))
        
        # Need to maximise the ELBO with respect to these weights:
        # - Generation network log_p_x_given_z --> Reconstruction loss
        # - KL-Divergence --> Latent loss between approximate posterior log_q_z and prior log_p_z
        loss = -tf.reduce_mean(log_p_x_given_z + log_p_z - log_q_z)
        tf.summary.scalar('elbo', -loss)

        return loss


def create_vae(
    data_size,
    latent_size,
    mixture_components=1,
    fcnet_hidden_sizes=None,
    hidden_activation_fn=tf.nn.relu,
    sigma_min=0.001,
    raw_sigma_bias=0.25,
    random_seed=None):
    """A factory method for creating VAEs.

    Args:
        data_size: The dimension of the vector that makes up the flattened images.
        latent_size: The size of the stochastic latent state of the GMVAE.
        mixture_components: The number of components in the mixture prior distribution.
            Defaults to 1 if not learning the prior parameters and using a single Gaussian.
        fcnet_hidden_sizes: A list of python integers, the size of the hidden
            layers of the fully connected networks that parameterise the conditional
            distributions of the VAE. If None, then it defaults to one hidden
            layer of size latent_size.
        hidden_activation_fn: The activation operation applied to intermediate 
            layers, and optionally to the output of the final layer.
        sigma_min: The minimum value that the standard deviation of the
            distribution over the latent state can take.
        raw_sigma_bias: A scalar that is added to the raw standard deviation
            output from the neural networks that parameterise the prior and
            approximate posterior. Useful for preventing standard deviations close to zero.
        random_seed: A random seed for the resampling operations.

    Returns:
        model: A TrainableVAE object.
    """

    if fcnet_hidden_sizes is None:
        fcnet_hidden_sizes = [latent_size]

    if mixture_components > 1:
        # A Gaussian mixture prior where parameters are learnt
        loc = tf.get_variable(
            name='loc', shape=[mixture_components, latent_size])
        raw_scale_diag = tf.get_variable(
            name='raw_scale_diag', shape=[mixture_components, latent_size])
        mixture_logits = tf.get_variable(
            name='mixture_logits', shape=[mixture_components])

        prior = tfp.distributions.MixtureSameFamily(
            components_distribution=tfp.distributions.MultivariateNormalDiag(
                loc=loc, scale_diag=tf.nn.softplus(raw_scale_diag)),
            mixture_distribution=tfp.distributions.Categorical(logits=mixture_logits),
            name='prior')
    else:
        # The prior, a multivariate Gaussian implementing p(z)
        prior = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros([latent_size]),
            scale_identity_multiplier=1.0,
            name='prior')

    # The generative distribution p(x | z) is conditioned on the latent
    # state variable z via a fully connected network
    decoder = base.ConditionalBernoulli(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        name='decoder')

    # A callable that implements the inference distribution q(z | x)
    encoder = base.ConditionalNormal(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        hidden_activation_fn=hidden_activation_fn,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        name='encoder')

    return TrainableVAE(prior, decoder, encoder, mixture_components, random_seed=random_seed)