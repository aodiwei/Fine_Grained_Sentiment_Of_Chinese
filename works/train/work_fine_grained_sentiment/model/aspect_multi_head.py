#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/10/21'
# 
"""
import tensorflow as tf

from works.train.base.model_base import ModelBase
from works.train.work_fine_grained_sentiment.model.multi_head import multihead


class AspectMultiHead(ModelBase):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 # batch_size,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 num_layer,
                 l2_reg_lambda=0.0001):
        super().__init__()
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.hidden_size = hidden_size
        self.set_input(input_x_shape=[None, sequence_length], input_y_shape=[None, 80],
                       input_x_dtype=tf.int32, input_y_dtype=tf.float32)
        self.l2_reg_lambda = l2_reg_lambda

        with tf.variable_scope("embedding_projection"), tf.device('/cpu:0'):  # embedding matrix
            self.embedding_W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='embedding_W')
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding_W, self.input_x)

        input = self.embedded_chars
        for i in range(num_layer):
            input = multihead.multihead_attention(queries=input, keys=input,
                                                  scope='multihead_attention_{}'.format(i),
                                                  dropout_rate=self.dropout_keep_prob)
        # FFN(x) = LN(x + point-wisely NN(x))
        # outputs = multihead.feedforward(ma, [self.hidden_size, self.embed_size])

        outputs = tf.reshape(input, [-1, self.sequence_length * self.embed_size])
        # logits = tf.layers.dense(outputs, units=self.num_classes)

        last = tf.layers.dense(outputs, 512,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                               name='last_fc')

        y_indx = 0
        losses = tf.constant(0.0)
        predictions = []
        accs = []
        for i in range(20):
            with tf.name_scope('aspect_{}'.format(i)):
                end_out = last
                # end_out = tf.layers.dense(end_out, 128,
                #                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                           kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                #                           name='aspect_{}_fc'.format(i))

                score = tf.layers.dense(end_out, self.num_classes,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                                        name='aspect_{}_score'.format(i))

                prediction = tf.argmax(score, 1, name='aspect_{}_prediction'.format(i))
                predictions.append(prediction)
                labels = self.input_y[:, y_indx: y_indx + 4]
                acc = self.get_accuracy(prediction, labels, i)
                accs.append(acc)
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=labels, name='aspect_{}_loss'.format(i))
                loss = tf.reduce_mean(loss, name='{}_loss'.format(i))
                losses += loss
                y_indx += 4

        self.predictions = tf.convert_to_tensor(predictions, name='predictions')
        self.accuracy = tf.reduce_mean(tf.cast(accs, 'float'), name='accuracy')
        self.loss = losses

        if self.l2_reg_lambda > 0:
            self.add_regularization(all_var=True, lam=self.l2_reg_lambda)
