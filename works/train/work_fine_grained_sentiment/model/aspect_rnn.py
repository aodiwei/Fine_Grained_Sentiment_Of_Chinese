#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/6/25'
# 
"""
import tensorflow as tf

from works.train.base.model_base import ModelBase


class AspectRnn(ModelBase):
    def __init__(self,
                 sequence_length,
                 vocab_size,
                 embedding_size,
                 num_hidden,
                 num_layer,
                 num_classes,
                 l2_reg_lambda,
                 bidirectional,
                 cell_type='lstm',
                 atten=None,
                 atten_size=50,
                 pre_word=False,
                 special_class=None):
        """

        :param sequence_length:
        :param vocab_size:
        :param embedding_size:
        :param num_hidden:
        :param num_layer:
        :param num_classes:
        :param l2_reg_lambda:
        :param bidirectional:
        :param cell_type:
        """
        super().__init__()
        if special_class is None:
            special_class = []

        self.pre_word = pre_word
        self.special_class = special_class
        self.set_input(input_x_shape=[None, sequence_length], input_y_shape=[None, 80],
                       input_x_dtype=tf.int32, input_y_dtype=tf.float32)

        last = self.build_lstm_seq(vocab_size, embedding_size, self.input_x,
                                   num_layer, num_hidden, bidirectional, 'content_seq', cell_type, atten)

        y_indx = 0
        losses = tf.constant(0.0)
        predictions = []
        accs = []
        for i in range(20):
            if self.special_class and i not in self.special_class:
                y_indx += 4
                print('class {} not in special class'.format(i))
                continue
            with tf.name_scope('aspect_{}'.format(i)):
                # fc_end = tf.layers.dense(fc_end, 1024,
                #                          activation=tf.nn.relu,
                #                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                #                          name='aspect{}_fc_1'.format(i))
                # fc_end = tf.layers.dropout(fc_end, self.dropout_keep_prob)
                end_out = last
                if atten and atten == 'hi':
                    with tf.name_scope('{}_hi_attention_layer'.format(i)):
                        end_out, alphas = self.hierarchical_attention(end_out, atten_size, return_alphas=True)
                        tf.summary.histogram('aspect_{}_attention_alphas'.format(i), alphas)
                        end_out = tf.nn.dropout(end_out, self.dropout_keep_prob)
                elif atten and atten == 'bi':
                    with tf.name_scope('{}_bi_attention_layer'.format(i)):
                        end_out = self.bi_attention(end_out, num_hidden, sequence_length)
                        end_out = tf.nn.dropout(end_out, self.dropout_keep_prob)

                score = tf.layers.dense(end_out, num_classes,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                                        name='aspect{}_fc_2'.format(i))

                prediction = tf.argmax(score, 1, name='aspect_{}_prediction'.format(i))
                predictions.append(prediction)
                labels = self.input_y[:, y_indx: y_indx + 4]
                acc = self.get_accuracy(prediction, labels, i)
                accs.append(acc)
                # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=score, labels=labels, name='aspect_{}_loss'.format(i))
                loss = self.focal_loss_softmax(score, labels)
                # loss = self.focal_loss(score, labels)
                loss = tf.reduce_mean(loss, name='{}_loss'.format(i))
                losses += loss

                y_indx += 4

        self.predictions = tf.convert_to_tensor(predictions, name='predictions')
        self.accuracy = tf.reduce_mean(tf.cast(accs, 'float'), name='accuracy')
        self.loss = losses

        if l2_reg_lambda > 0:
            self.add_regularization(all_var=True, lam=l2_reg_lambda)

    def build_lstm_seq(self, vocab_size, embed_dim, input_x, num_layers, num_hidden, bidirectional, name, cell_type, atten=None):
        """

        :param vocab_size:
        :param embed_dim:
        :param input_x:
        :param num_layers:
        :param num_hidden:
        :param bidirectional:
        :param name:
        :param cell_type:
        :param atten:
        :return:
        """
        with tf.name_scope(name):
            with tf.name_scope('embedding'), tf.device('/cpu:0'):
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embed_dim], -1.0, 1.0), name='W',
                    trainable=not self.pre_word
                )
                print('>>>pre word flag {}'.format(self.pre_word))
                if self.pre_word:
                    print('>>>use pre word')
                    self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embed_dim])
                    self.embedding_init = W.assign(self.embedding_placeholder)

                embedded_input_x = tf.nn.embedding_lookup(W, input_x)

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
                                                                             inputs=embedded_input_x, dtype=tf.float32)
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
                    outputs, output_states = tf.nn.dynamic_rnn(rnn_cells, embedded_input_x, dtype=tf.float32)
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
