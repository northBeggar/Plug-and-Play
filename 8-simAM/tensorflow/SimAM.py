#!/usr/bin/env python
# coding: utf-8 
import tensorflow as tf
 
class SimAM(tf.keras.layers.Layer):
    def __init__(self, eps=1e-7, activaton=tf.nn.sigmoid, trainable=True, name=None, **kwargs):
        super(SimAM, self).__init__(name=name, trainable=trainable, **kwargs)
        self.activaton = activaton
        self.eps = eps

    def build(self, input_shape):
        if isinstance(input_shape[-1], int):
            self.height = input_shape[1]
            self.width = input_shape[2]
            self.channels = input_shape[3]
        else:
            self.height = input_shape[1].value
            self.width = input_shape[2].value
            self.channels = input_shape[3].value
        self.norm = 4. / (self.height * self.width - 1)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        minus_mu_square = tf.math.square(inputs - tf.math.reduce_mean(inputs, axis=(1, 2), keepdims=True))
        out = minus_mu_square / tf.maximum(
            tf.math.reduce_sum(minus_mu_square, axis=(1, 2), keepdims=True) * self.norm,
            self.eps) + 0.5
        return inputs * self.activaton(out)
