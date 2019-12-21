from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions
DEFAULT_INITIALIZERS = {'w': tf.contrib.layers.xavier_initializer(), 'b': tf.zeros_initializer()}


class ConditionalNormal(object):
    """A MultivariateNormalDiag distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes=None, initializers=DEFAULT_INITIALIZERS, 
                 sigma_min=0.0, raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu, 
                 name='cond_normal'):
        """Creates a conditional MultivariateNormalDiag distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs. If None,
                then the default is a single-layered dense network.
            initializers: The variable initializers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initializers for
                the weights and biases.
            sigma_min: The minimum standard deviation allowed, a scalar.
            raw_sigma_bias: A scalar that is added to the raw standard deviation
                output from the fully connected network. Set to 0.25 by default to
                prevent standard deviations close to 0.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self._name = name
        self._size = size
        self._sigma_min = sigma_min
        self._raw_sigma_bias = raw_sigma_bias

        if hidden_layer_sizes is not None:
            self._fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [2*size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')
        else:
            self._fcnet = snt.nets.MLP(
                output_sizes=[2*size],
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the parameters of a MultivariateNormalDiag distribution based on the inputs."""

        inputs = tf.concat(tensor_list, axis=1)
        outs = self._fcnet(inputs)

        mu, sigma = tf.split(outs, 2, axis=1)
        sigma = tf.maximum(tf.nn.softplus(sigma + self._raw_sigma_bias), self._sigma_min)
        
        return mu, sigma


    def __call__(self, *args, **kwargs):
        """Creates a MultivariateNormalDiag distribution conditioned on the inputs."""

        mu, sigma = self.condition(args, **kwargs)
        
        return tfd.MultivariateNormalDiag(
            loc=mu, 
            scale_diag=sigma, 
            name=self._name)


class ConditionalBernoulli(object):
    """A Bernoulli distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes=None, initializers=DEFAULT_INITIALIZERS, 
                 bias_init=0.0, hidden_activation_fn=tf.nn.relu, name='cond_bernoulli'):
        """Creates a conditional Bernoulli distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs. If None,
                then the default is a single-layered dense network.
            initializers: The variable initializers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initializers for
                the weights and biases.
            bias_init: A scalar or vector Tensor that is added to the output of the
                fully connected network and parameterizes the distribution mean.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self._name = name
        self._size = size
        self._bias_init = bias_init

        if hidden_layer_sizes is not None:
            self._fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')
        else:
            self._fcnet = snt.nets.MLP(
                output_sizes=[size],
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the logits of a Bernoulli distribution."""

        inputs = tf.concat(tensor_list, axis=1)
        
        return self._fcnet(inputs) + self._bias_init


    def __call__(self, *args, **kwargs):
        """Creates a Bernoulli distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)
        
        return tfd.Independent(
            tfd.Bernoulli(logits=logits),
            reinterpreted_batch_ndims=1, # Assuming 1-D vector inputs (bs discluded)
            name=self._name)


class ConditionalCategorical(object):
    """A RelaxedOneHotCategorical distribution conditioned on Tensor inputs via a dense network."""

    def __init__(self, size, hidden_layer_sizes=None, temperature=1.0, initializers=DEFAULT_INITIALIZERS, 
                 hidden_activation_fn=tf.nn.relu, name='cond_categorical'):
        """Creates a conditional RelaxedOneHotCategorical distribution.

        Args:
            size: The dimension of the random variable.
            hidden_layer_sizes: The sizes of the hidden layers of the fully connected
                network used to condition the distribution on the inputs. If None,
                then the default is a single-layered dense network.
            temperature: Degree of how approximately discrete the distribution is. The closer 
                to 0, the more discrete and the closer to infinity, the more uniform.
            initializers: The variable initializers to use for the fully connected
                network. The network is implemented using snt.nets.MLP so it must
                be a dictionary mapping the keys 'w' and 'b' to the initializers for
                the weights and biases.
            hidden_activation_fn: The activation function to use on the hidden layers
                of the fully connected network.
            name: The name of this distribution, used for sonnet scoping.
        """

        self._name = name
        self._size = size
        self._temperature = temperature

        if hidden_layer_sizes is not None:
            self._fcnet = snt.nets.MLP(
                output_sizes=hidden_layer_sizes + [size],
                activation=hidden_activation_fn,
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')
        else:
            self._fcnet = snt.nets.MLP(
                output_sizes=[size],
                initializers=initializers,
                activate_final=False,
                use_bias=True,
                name=name + '_fcnet')


    def condition(self, tensor_list, **unused_kwargs):
        """Computes the logits of a RelaxedOneHotCategorical distribution."""

        inputs = tf.concat(tensor_list, axis=1)
        
        return self._fcnet(inputs)


    def __call__(self, *args, **kwargs):
        """Creates a RelaxedOneHotCategorical distribution conditioned on the inputs."""

        logits = self.condition(args, **kwargs)
        
        return tfd.RelaxedOneHotCategorical(
            self._temperature, 
            logits=logits, 
            name=self._name)