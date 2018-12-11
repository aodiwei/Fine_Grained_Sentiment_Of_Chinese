#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/10/16'
# 
"""
import tensorflow as tf

from works.train.base.model_base import ModelBase


class AspectCnnRnn(ModelBase):

    def __init__(self,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 filter_sizes,
                 num_filters,
                 num_hidden,
                 num_layer,
                 num_classes,
                 l2_reg_lambda,
                 bidirectional,
                 cell_type='lstm',
                 atten=None,
                 atten_size=50,

                 ):
        super().__init__()
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_layer = num_layer
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        self.atten = atten
        self.atten_size = atten_size

        self.set_input(input_x_shape=[None, self.sequence_length], input_y_shape=[None, 80],
                       input_x_dtype=tf.int32, input_y_dtype=tf.float32)

        self.build_model()

    def build_embedding(self):
        """

        :return:
        """
        with tf.name_scope('embedding'):
            with tf.device('/cpu:0'):
                self.embedding_W = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name='embedding_W')
                embedded_chars = tf.nn.embedding_lookup(self.embedding_W, self.input_x)
                embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        return embedded_chars, embedded_chars_expanded

    def build_model(self):
        """

        :return:
        """
        embedded_chars, embedded_chars_expanded = self.build_embedding()
        rnn_last = self.build_lstm_seq(embedded_chars, self.num_layer, self.num_hidden, self.bidirectional, 'content_seq', self.cell_type,
                                       self.atten)
        # cnn_last = self.build_cnn(embedded_chars_expanded)

        y_indx = 0
        losses = tf.constant(0.0)
        predictions = []
        accs = []
        for i in range(20):
            with tf.name_scope('aspect_{}'.format(i)):
                end_out = rnn_last
                if self.atten and self.atten == 'hi':
                    with tf.name_scope('{}_hi_attention_layer'.format(i)):
                        end_out, alphas = self.hierarchical_attention(end_out, self.atten_size, return_alphas=True)
                        tf.summary.histogram('aspect_{}_attention_alphas'.format(i), alphas)
                        end_out = tf.nn.dropout(end_out, self.dropout_keep_prob)
                elif self.atten and self.atten == 'bi':
                    with tf.name_scope('{}_bi_attention_layer'.format(i)):
                        end_out = self.bi_attention(end_out, self.num_hidden, self.sequence_length)
                        end_out = tf.nn.dropout(end_out, self.dropout_keep_prob)

                with tf.name_scope('concat_cnn'):
                    cnn_last = self.build_cnn(embedded_chars_expanded)
                    end_out = tf.concat([end_out, cnn_last], axis=1)
                    end_out = tf.layers.dropout(end_out, self.dropout_keep_prob)

                score = tf.layers.dense(end_out, self.num_classes,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                                        name='aspect{}_fc_2'.format(i))

                prediction = tf.argmax(score, 1, name='aspect_{}_prediction'.format(i))
                predictions.append(prediction)
                labels = self.input_y[:, y_indx: y_indx + 4]
                acc = self.get_accuracy(prediction, labels, i)
                accs.append(acc)
                # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=labels, name='aspect_{}_loss'.format(i))
                loss = self.focal_loss_softmax(score, labels)
                loss = tf.reduce_mean(loss, name='{}_loss'.format(i))
                losses += loss

                y_indx += 4

        self.predictions = tf.convert_to_tensor(predictions, name='predictions')
        self.accuracy = tf.reduce_mean(tf.cast(accs, 'float'), name='accuracy')
        self.loss = losses

        if self.l2_reg_lambda > 0:
            self.add_regularization(all_var=False, lam=self.l2_reg_lambda)

    def build_cnn(self, embedded_chars_expanded):
        """

        :return:
        """
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # Convolution
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('vec_dropout'):
            h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        return h_drop

    def build_lstm_seq(self, embedded_chars, num_layers, num_hidden, bidirectional, name, cell_type, atten=None):
        """

        :param embedded_chars:
        :param num_layers:
        :param num_hidden:
        :param bidirectional:
        :param name:
        :param cell_type:
        :param atten:
        :return:
        """
        with tf.variable_scope(name + '_rnn', initializer=tf.orthogonal_initializer()):
            def build():
                cells = []
                for _ in range(num_layers):
                    if cell_type == 'lstm':
                        basic_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

                    else:
                        basic_cell = tf.nn.rnn_cell.GRUCell(num_hidden, kernel_initializer=tf.orthogonal_initializer())
                    if not bidirectional:
                        basic_cell = tf.nn.rnn_cell.DropoutWrapper(basic_cell, output_keep_prob=self.dropout_keep_prob)
                    cells.append(basic_cell)
                # 多层
                stacked_rnn = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
                return stacked_rnn

            if bidirectional:
                fw_rnn_cells = build()
                bw_rnn_cells = build()
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_rnn_cells, cell_bw=bw_rnn_cells,
                                                                         inputs=embedded_chars, dtype=tf.float32)
                if atten:
                    if atten == 'hi':
                        last = outputs
                    elif atten == 'bi':
                        fw_outputs, bw_outputs = outputs
                        last = fw_outputs + bw_outputs
                else:
                    last = tf.concat((outputs[0][:, -1], outputs[1][:, -1]), axis=1)
            else:
                rnn_cells = build()
                # stacked_lstm = tf.nn.rnn_cell.MultiRNNCell(rnn_cells, state_is_tuple=True)
                outputs, output_states = tf.nn.dynamic_rnn(rnn_cells, embedded_chars, dtype=tf.float32)
                if atten:
                    last = outputs
                else:
                    val = tf.transpose(outputs, [1, 0, 2])
                    last = tf.gather(val, int(val.get_shape()[0]) - 1)
            return last

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
