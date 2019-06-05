# GMVAE

This repository contains a TensorFlow implementation of a Gaussian Mixture Variational Autoencoder (GMVAE) on the MNIST dataset, specifically making use of the [Probability](https://www.tensorflow.org/probability) library.

There are currently three models in use:
- **VAE** is a standard implementation of the [Variational Autoencoder](https://arxiv.org/abs/1312.6114), with no convolutional layers  
- **VAE_GMP** is an adaptation of VAE to make use of a Gaussian Mixture prior, instead of a standard Normal distribution
- **GMVAE** is an attempt to replicate the work described in this [blog](http://ruishu.io/2016/12/25/gmvae/) and inspired from this [paper](https://arxiv.org/abs/1611.02648)

The directory layout is as follows:
- `bin`: Bash example scripts for running the aforementioned models
- `checkpoints`: Directories to save checkpoints of trained model states
- `scripts`: TensorFlow scripts to implement the models and run them using the main `run_gmvae.py` script, alongside other helpful modules (`helpers.py` and `base.py`)

*Note:* This is a work in progress, so any contributions/feedback will be well-received.

## Dependencies

- TensorFlow 1.13.1
- [TensorFlow Datasets](https://github.com/tensorflow/datasets)
- TensorFlow Probability 0.6.0
- Cuda 10.0
- Cudnn 7.4.2
