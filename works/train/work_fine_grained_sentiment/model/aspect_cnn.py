#!/usr/bin/env python
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/9/16'
#
"""
import tensorflow as tf

from works.train.base.model_base import ModelBase


class AspectCnn(ModelBase):
    """

    """

    def __init__(self,
                 sequence_length,
                 num_classes,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 l2_reg_lambda=0.0):
        super().__init__()
        self.set_input(input_x_shape=[None, sequence_length], input_y_shape=[None, 80],
                       input_x_dtype=tf.int32, input_y_dtype=tf.float32)

        # Embedding
        with tf.name_scope('embedding'):
            with tf.device('/cpu:0'):
                self.embedding_W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='embedding_W')
                self.embedded_chars = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('vec_dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        y_indx = 0
        losses = tf.constant(0.0)
        predictions = []
        accs = []
        for i in range(20):
            with tf.name_scope('aspect_{}'.format(i)):
                # W = tf.get_variable(
                #     'aspect_{}_W'.format(i),
                #     shape=[num_filters_total, num_classes],
                #     initializer=tf.contrib.layers.xavier_initializer())
                # b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')

                fc_end = tf.layers.dense(self.h_drop, 512,
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                         name='aspect{}_fc_1'.format(i))
                fc_end = tf.layers.dropout(fc_end, self.dropout_keep_prob)

                score = tf.layers.dense(fc_end, num_classes,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                        name='aspect{}_fc_2'.format(i))

                prediction = tf.argmax(score, 1, name='aspect_{}_prediction'.format(i))
                predictions.append(prediction)
                labels = self.input_y[:, y_indx: y_indx + 4]
                acc = self.get_accuracy(prediction, labels, i)
                accs.append(acc)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=labels, name='aspect_{}_loss'.format(i))
                # loss = self.focal_loss_softmax(score, labels)
                loss = tf.reduce_mean(loss, name='loss')

                losses += loss

                y_indx += 4

        self.predictions = tf.convert_to_tensor(predictions, name='predictions')
        self.accuracy = tf.reduce_mean(tf.cast(accs, 'float'), name='accuracy')
        self.loss = losses
        if l2_reg_lambda > 0:
            self.add_regularization(all_var=True, lam=l2_reg_lambda)

    def get_accuracy(self, prediction, labels, aspect_idx):
        """

        :param prediction:
        :param labels:
        :param aspect_idx:
        :return:
        """
        correct_predictions = tf.equal(prediction, tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='{}_accuracy'.format(aspect_idx))

        return accuracy
