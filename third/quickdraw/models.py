# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import tensorflow as tf
import utils

import tensorflow.contrib.slim as slim

"""Contains the base class for models."""
class BaseModel(object):
  """Inherit from this class when implementing new models."""

  def create_model(self, unused_model_input, **unused_params):
    raise NotImplementedError()

class LogisticModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      num_classes: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    net = slim.flatten(model_input)
    output = slim.fully_connected(
        net, num_classes, activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class myModel(BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, num_classes=10, p_keep_prob, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      num_classes: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""


    x=slim.flatten(model_input)
    W=tf.Variable(tf.zeros([2500,10]))
    b=tf.Variable(tf.zeros([10]))

    #

    #avoid gradient 0 
    def weight_variable(shape):
      initial=tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')



    #layer1 conv1
    x_image = tf.reshape(x, [-1,50,50,1])
    # window 크기, input channel, output(=num of feat or filters)
    W_conv1_1 = weight_variable([3, 3, 1, 16])
    W_conv1_2 = weight_variable([3, 3, 16, 32])
    b_conv1_1 = bias_variable([16]) 
    b_conv1_2 = bias_variable([32]) 

    h_conv1_1 = tf.nn.relu(conv2d(x_image, W_conv1_1) + b_conv1_1)
    h_conv1_2 = tf.nn.relu(conv2d(h_conv1_1, W_conv1_2) + b_conv1_2)
    h_pool1 = max_pool_2x2(h_conv1_2)
    #50*50->25*25

    #layer2 conv2
    W_conv2_1 = weight_variable([3, 3, 32, 64])
    W_conv2_2 = weight_variable([3, 3, 64, 64])
    b_conv2_1 = bias_variable([64])
    b_conv2_2 = bias_variable([64])

    h_conv2_1 = tf.nn.relu(conv2d(h_pool1, W_conv2_1) + b_conv2_1)
    h_conv2_2 = tf.nn.relu(conv2d(h_conv2_1, W_conv2_2) + b_conv2_2)
    h_pool2 = max_pool_2x2(h_conv2_2)
    #25*25->13*13

    #full connected
    h_pool2_flat = tf.reshape(h_pool2, [-1, 13*13*64])
    W_fc1 = weight_variable([13 * 13 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=p_keep_prob)

    # output
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    output=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return {"predictions": output}
