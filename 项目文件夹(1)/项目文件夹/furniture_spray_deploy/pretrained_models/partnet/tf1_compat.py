"""Patch tf.contrib.layers for TF2 compatibility.

PointCNN uses tf.contrib.layers.l2_regularizer and
tf.contrib.layers.separable_conv2d which were removed in TF2.
This module monkey-patches them before the PointCNN code imports.
Import this BEFORE any PointCNN module.
"""

import tensorflow as tf
import numpy as np


def _l2_regularizer(scale=1.0):
    """Replacement for tf.contrib.layers.l2_regularizer.
    During inference the regularizer loss is never used, so return no-op."""
    return lambda x: tf.constant(0.0)


def _separable_conv2d(inputs, num_outputs, kernel_size,
                      depth_multiplier=1, stride=1, padding='VALID',
                      activation_fn=tf.nn.relu,
                      weights_initializer=None,
                      weights_regularizer=None,
                      biases_initializer=None,
                      biases_regularizer=None,
                      reuse=None, scope=None, trainable=True):
    """Replacement for tf.contrib.layers.separable_conv2d.
    Reimplemented using tf.compat.v1 APIs available in TF2."""
    in_channels = inputs.shape[-1]
    if in_channels is None:
        in_channels = tf.shape(inputs)[-1]
    kernel_h, kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    if num_outputs is None:
        # Depthwise convolution only
        depthwise_filter = tf.compat.v1.get_variable(
            name=scope + '/depthwise_weights',
            shape=[kernel_h, kernel_w, in_channels, depth_multiplier],
            initializer=weights_initializer or tf.compat.v1.glorot_normal_initializer(),
            regularizer=weights_regularizer)
        outputs = tf.nn.depthwise_conv2d(
            inputs, depthwise_filter,
            strides=[1, stride, stride, 1],
            padding=padding)
    else:
        # Full separable conv: depthwise + pointwise
        depthwise_filter = tf.compat.v1.get_variable(
            name=scope + '/depthwise_weights',
            shape=[kernel_h, kernel_w, in_channels, depth_multiplier],
            initializer=weights_initializer or tf.compat.v1.glorot_normal_initializer(),
            regularizer=weights_regularizer)
        pointwise_filter = tf.compat.v1.get_variable(
            name=scope + '/pointwise_weights',
            shape=[1, 1, in_channels * depth_multiplier, num_outputs],
            initializer=weights_initializer or tf.compat.v1.glorot_normal_initializer(),
            regularizer=weights_regularizer)
        outputs = tf.nn.separable_conv2d(
            inputs, depthwise_filter, pointwise_filter,
            strides=[1, stride, stride, 1],
            padding=padding)

    if biases_initializer is not None:
        out_channels = outputs.shape[-1]
        if out_channels is None:
            out_channels = num_outputs if num_outputs is not None else in_channels * depth_multiplier
        biases = tf.compat.v1.get_variable(
            name=scope + '/biases',
            shape=[out_channels],
            initializer=biases_initializer,
            regularizer=biases_regularizer)
        outputs = tf.nn.bias_add(outputs, biases)

    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs


class _CompatLayers:
    l2_regularizer = staticmethod(_l2_regularizer)
    separable_conv2d = staticmethod(_separable_conv2d)


class _Contrib:
    layers = _CompatLayers


# Apply contrib patch
tf.contrib = _Contrib()

# Also restore tf.layers which was removed in TF 2.x
if not hasattr(tf, 'layers'):
    tf.layers = tf.compat.v1.layers

# And the glorot initializer alias
tf.glorot_normal_initializer = tf.compat.v1.glorot_normal_initializer

# tf.random_normal -> compat
tf.random_normal = tf.compat.v1.random_normal

# tf.zeros_initializer -> compat
tf.zeros_initializer = tf.compat.v1.zeros_initializer
