from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
DEFAULT_INITIALISERS = {'w': tf.contrib.layers.xavier_initializer(), 'b': tf.zeros_initializer()}


class GaussianMixture(object):
    """A Gaussian Mixture Model conditioned on Tensor inputs via dense networks."""

    def __init__(self, size, hidden_layer_sizes, initialisers=DEFAULT_INITIALISERS, 
                 mixture_components=1, sigma_min=0.0, raw_sigma_bias=0.25, 
                 hidden_activation_fn=tf.nn.relu, name='gaussian_mixture'):
        """Creates a GMM that is conditioned on a dense network to learn the parameters
        of the component distribution (multivariate Gaussian).

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the mixture on the inputs.
            initialisers: The variable initialisers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initialisers for
                the weights and biases.
            mixture_components: The number of components in the mixture.
            sigma_min: The minimum standard deviation allowed, a scalar.
            raw_sigma_bias: A scalar that is added to the raw standard deviation
                output from the fully connected network. Set to 0.25 by default to
                prevent standard deviations close to 0.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.name = name
        self.size = size
        self.mix_components = mixture_components

        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [mixture_components*2*size],
            activation=hidden_activation_fn,
            initializers=initialisers,
            activate_final=False,
            use_bias=True,
            name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a MultivariateNormalDiag distribution based on the inputs."""

        inputs = tf.concat(tensor_list, axis=1)
        outs = self.fcnet(inputs)

        mu, sigma = tf.split(outs, 2, axis=1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)

        mu = tf.reshape(mu, [-1, self.mix_components, self.size])
        sigma = tf.reshape(sigma, [-1, self.mix_components, self.size])

        return mu, sigma


    def __call__(self, *args, **kwargs):
        """Creates a GMM distribution conditioned on the inputs."""

        mu, sigma = self.condition(args, **kwargs)

        if 'cat_logits' not in kwargs:
            tf.logging.error("No categorical variables provided: %s", kwargs.get('cat_logits'))
            return None

        return tfd.MixtureSameFamily(
            components_distribution=tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma),
            mixture_distribution=tfd.Categorical(logits=kwargs.get('cat_logits')),
            name=self.name)


class ConditionalNormal(object):
    """A MultivariateNormalDiag distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes, initialisers=DEFAULT_INITIALISERS, 
                 sigma_min=0.0, raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu, 
                 name='conditional_normal'):
        """Creates a conditional MultivariateNormalDiag distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs.
            initialisers: The variable initialisers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initialisers for
                the weights and biases.
            sigma_min: The minimum standard deviation allowed, a scalar.
            raw_sigma_bias: A scalar that is added to the raw standard deviation
                output from the fully connected network. Set to 0.25 by default to
                prevent standard deviations close to 0.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self.sigma_min = sigma_min
        self.raw_sigma_bias = raw_sigma_bias
        self.name = name
        self.size = size

        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [2*size],
            activation=hidden_activation_fn,
            initializers=initialisers,
            activate_final=False,
            use_bias=True,
            name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a MultivariateNormalDiag distribution based on the inputs."""

        inputs = tf.concat(tensor_list, axis=1)
        outs = self.fcnet(inputs)

        mu, sigma = tf.split(outs, 2, axis=1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias), self.sigma_min)
        
        return mu, sigma


    def __call__(self, *args, **kwargs):
        """Creates a MultivariateNormalDiag distribution conditioned on the inputs."""

        mu, sigma = self.condition(args, **kwargs)
        
        return tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma, name=self.name)


class ConditionalBernoulli(object):
    """A Bernoulli distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes, initialisers=DEFAULT_INITIALISERS, 
                 hidden_activation_fn=tf.nn.relu, name='conditional_bernoulli'):
        """Creates a conditional Bernoulli distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs.
            initialisers: The variable initialisers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initialisers for
                the weights and biases.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self.name = name
        self.size = size

        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [size],
            activation=hidden_activation_fn,
            initializers=initialisers,
            activate_final=False,
            use_bias=True,
            name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a Bernoulli distribution based on the inputs."""

        inputs = tf.concat(tensor_list, axis=1)
        logits = self.fcnet(inputs)
        
        return logits


    def __call__(self, *args, **kwargs):
        """Creates a Bernoulli distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)
        
        return tfd.Independent(
            tfd.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=1, # Assuming 1-D vector inputs (bs discluded)
            name=self.name)


class ConditionalCategorical(object):
    """A Categorical distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes, initialisers=DEFAULT_INITIALISERS, 
                 hidden_activation_fn=tf.nn.relu, name='conditional_categorical'):
        """Creates a conditional Categorical distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs.
            initialisers: The variable initializers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initialisers for
                the weights and biases.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self.name = name
        self.size = size

        self.fcnet = snt.nets.MLP(
            output_sizes=hidden_layer_sizes + [size],
            activation=hidden_activation_fn,
            initializers=initialisers,
            activate_final=False,
            use_bias=True,
            name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a Categorical distribution based on the inputs."""

        inputs = tf.concat(tensor_list, axis=1)
        logits = self.fcnet(inputs)
        
        return logits


    def __call__(self, *args, **kwargs):
        """Creates a Categorical distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)
        
        return tfd.Categorical(logits=logits, name=self.name)