#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/10/17'
# 
"""
import tensorflow as tf

from works.train.base.model_base import ModelBase
from works.train.work_fine_grained_sentiment.model.transformer.encoder import Encoder
from works.train.work_fine_grained_sentiment.model.transformer.transformer_base_model import TransformerBaseClass


class AspectTransformer(ModelBase, TransformerBaseClass):
    def __init__(self,
                 sequence_length,
                 num_classes,
                 batch_size,
                 vocab_size,
                 embedding_size,
                 d_model,
                 d_k,
                 d_v,
                 h,
                 num_layer,
                 l2_reg_lambda=0.0001,
                 use_residual_conn=False
                 ):
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        # self.learning_rate = tf.Variable(learning_rate, trainable=False, name="learning_rate")
        # self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * 0.5)
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.l2_reg_lambda = l2_reg_lambda

        self.use_residual_conn = use_residual_conn

        # 不要一说到 super 就想到父类！super 指的是 MRO 中的下一个类！
        super().__init__()
        super(ModelBase, self).__init__(d_model, d_k, d_v, sequence_length, h, batch_size, num_layer=num_layer)

        self.set_input(input_x_shape=[None, sequence_length], input_y_shape=[None, 80],
                       input_x_dtype=tf.int32, input_y_dtype=tf.float32)
        self.instantiate_weights()
        self.build_model()

    def build_model(self):
        """

        :return:
        """
        last = self.inference()
        y_indx = 0
        losses = tf.constant(0.0)
        predictions = []
        accs = []
        for i in range(20):
            with tf.name_scope('aspect_{}'.format(i)):
                end_out = last
                W_projection = tf.get_variable('w_projection_{}'.format(i), shape=[self.sequence_length * self.d_model, self.num_classes],
                                               initializer=self.initializer)
                b_projection = tf.get_variable('b_projection_{}'.format(i), shape=[self.num_classes])

                score = tf.add(tf.matmul(end_out, W_projection), b_projection, name='aspect{}_score'.format(i))

                # score = tf.layers.dense(end_out, self.num_classes,
                #                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                #                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda),
                #                         name='aspect{}_fc_2'.format(i))

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

    def instantiate_weights(self):
        """define all weights here"""
        with tf.variable_scope("embedding_projection"), tf.device('/cpu:0'):  # embedding matrix
            self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size],
                                             initializer=self.initializer)
            # self.Embedding_label = tf.get_variable("Embedding_label", shape=[self.num_classes, self.embed_size],
            #                                        dtype=tf.float32)  # ,initializer=self.initializer
            # self.W_projection = tf.get_variable("W_projection", shape=[self.sequence_length * self.d_model, self.num_classes],
            #                                     initializer=self.initializer)  # [embed_size,label_size]
            # self.b_projection = tf.get_variable("b_projection", shape=[self.num_classes])

    def get_mask(self, sequence_length):
        lower_triangle = tf.matrix_band_part(tf.ones([sequence_length, sequence_length]), -1, 0)
        result = -1e9 * (1.0 - lower_triangle)
        print("get_mask==>result:", result)
        return result

    def inference(self):
        """ building blocks:
        encoder:6 layers.each layers has two   sub-layers. the first is multi-head self-attention mechanism; the second is position-wise fully connected feed-forward network.
               for each sublayer. use LayerNorm(x+Sublayer(x)). all dimension=512.
        decoder:6 layers.each layers has three sub-layers. the second layer is performs multi-head attention over the ouput of the encoder stack.
               for each sublayer. use LayerNorm(x+Sublayer(x)). 分类不decoder
        """
        # 1.embedding for encoder input & decoder input
        # 1.1 position embedding for encoder input
        input_x_embeded = tf.nn.embedding_lookup(self.Embedding, self.input_x)  # [None,sequence_length, embed_size]
        input_x_embeded = tf.multiply(input_x_embeded, tf.sqrt(tf.cast(self.d_model, dtype=tf.float32)))
        input_mask = tf.get_variable("input_mask", [self.sequence_length, 1], initializer=self.initializer)
        input_x_embeded = tf.add(input_x_embeded, input_mask)  # [None,sequence_length,embed_size].position embedding.

        # 2. encoder
        encoder_class = Encoder(self.d_model, self.d_k, self.d_v, self.sequence_length, self.h, self.batch_size, self.num_layer,
                                input_x_embeded, input_x_embeded, dropout_keep_prob=self.dropout_keep_prob,
                                use_residual_conn=self.use_residual_conn)
        Q_encoded, K_encoded = encoder_class.encoder_fn()  # K_v_encoder

        Q_encoded = tf.reshape(Q_encoded, shape=(self.batch_size, -1))  # [batch_size,sequence_length*d_model]
        # with tf.variable_scope("output"):
        #     logits = tf.matmul(Q_encoded,
        #                        self.W_projection) + self.b_projection  # logits shape:[batch_size*decoder_sent_length,self.num_classes]
        # print("logits:", logits)
        return Q_encoded
