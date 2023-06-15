#!/usr/bin/env python
#coding: utf-8

import tensorflow as tf
import tensorflow.contrib.slim as slim


def CoordAtt(x, reduction = 32):

    def coord_act(x):
        tmpx = tf.nn.relu6(x+3) / 6
        x = x * tmpx
        return x

    x_shape = x.get_shape().as_list()
    [b, h, w, c] = x_shape
    x_h = slim.avg_pool2d(x, kernel_size = [1, w], stride = 1)
    x_w = slim.avg_pool2d(x, kernel_size = [h, 1], stride = 1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])

    y = tf.concat([x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = slim.conv2d(y, mip, (1, 1), stride=1, padding='VALID', normalizer_fn = slim.batch_norm, activation_fn=coord_act,scope='ca_conv1')

    x_h, x_w = tf.split(y, num_or_size_splits=2, axis=1)
    x_w = tf.transpose(x_w, [0, 2, 1, 3])
    a_h = slim.conv2d(x_h, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv2')
    a_w = slim.conv2d(x_w, c, (1, 1), stride=1, padding='VALID', normalizer_fn = None, activation_fn=tf.nn.sigmoid,scope='ca_conv3')

    out = x * a_h * a_w


    return out















