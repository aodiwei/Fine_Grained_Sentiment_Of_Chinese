#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/12/27'
# 模型基类
"""
from abc import abstractmethod, ABCMeta

import tensorflow as tf
from tensorflow.python.ops import array_ops


class ModelBase(metaclass=ABCMeta):
    def __init__(self):
        self.loss = None
        self.accuracy = None
        self.input_x = None
        self.input_y = None
        self.dropout_keep_prob = None

    def set_input(self, input_x_shape, input_y_shape, input_x_dtype=tf.int32, input_y_dtype=tf.float32):
        """

        :param input_x_shape: 
        :param input_y_shape: 
        :param input_x_dtype: 
        :param input_y_dtype: 
        :return: 
        """
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(input_x_dtype, input_x_shape, name="input_x")
        self.input_y = tf.placeholder(input_y_dtype, input_y_shape, name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def set_predictions(self, scores):
        """
        准确数
        :param scores: 
        :return: 
        """
        self.predictions = tf.argmax(scores, 1, name='predictions')

    def set_loss(self, losses, reg=None):
        """
        :param losses: 
        :param reg: 正则化
        :return: 
        """
        if reg is None:
            self.loss = tf.reduce_mean(losses, name='loss')
        else:
            self.loss = tf.add(tf.reduce_mean(losses), reg, name='loss')

    def set_accuracy(self, predictions):
        """

        :param predictions: 
        :return: 
        """
        correct_predictions = tf.equal(predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

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

    def add_regularization(self, all_var=False, lam=0):
        """
        正则化
        :param all_var:
        :param lam:
        :return:
        """
        if all_var:
            tv = tf.trainable_variables()
            regularization_cost = tf.multiply(tf.reduce_sum([tf.nn.l2_loss(v) for v in tv if 'bias' not in v.name]), lam)
            self.loss += regularization_cost
        else:
            l2_loss = tf.losses.get_regularization_loss()
            self.loss += l2_loss

    def layers(self, hidden_layers_list, num_classes, l2_reg_lambda):
        """

        :param l2_reg_lambda: 
        :param num_classes: 
        :param hidden_layers_list: 
        :return: 
        """
        assert isinstance(hidden_layers_list, list), 'hidden_layers_list must be list'
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        net = tf.layers.dense(self.input_x, hidden_layers_list[0], activation=tf.nn.relu, kernel_initializer=kernel_initializer,
                              name='layer0')
        for i, lay in enumerate(hidden_layers_list[1:]):
            net = tf.layers.dense(net, lay, activation=tf.nn.relu, kernel_initializer=kernel_initializer,
                                  name='layer{}'.format(i + 1))
            net = tf.layers.dropout(net, rate=self.dropout_keep_prob, name='drop_{}'.format(i + 1))

        out = tf.layers.dense(net, num_classes, activation=tf.nn.relu, kernel_initializer=kernel_initializer,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg_lambda),
                              name='out')
        l2_loss = tf.losses.get_regularization_loss()
        return out, l2_loss

    def focal_loss(self, prediction_tensor, target_tensor, alpha=0.25, gamma=2):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        """
        sigmoid_p = tf.nn.sigmoid(prediction_tensor)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

        # For poitive prediction, only need consider front part loss, back part is 0;
        # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
        pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

        # For negative prediction, only need consider back part loss, front part is 0;
        # target_tensor > zeros <=> z=1, so negative coefficient = 0.
        neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                              - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
        return tf.reduce_sum(per_entry_cross_ent, name='focal_loss', axis=1)

    def focal_loss_softmax(self, score, onehot_labels, gamma=2.0, alpha=4.0):
        """
        focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: logits is probability after softmax tf.nn.softmax(logits) 这里传入还是score
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        Focal Loss for Dense Object Detection, 130(4), 485–491.
        https://doi.org/10.1016/j.ajodo.2005.02.022
        :param labels: ground truth labels, shape of [batch_size]
        :param logits: model's output, shape of [batch_size, num_cls]
        :param gamma:
        :param alpha:
        :return: shape of [batch_size]
        """
        epsilon = 1.e-9

        logits = tf.nn.softmax(score)

        model_out = tf.add(logits, epsilon)
        ce = tf.multiply(onehot_labels, -tf.log(model_out))
        weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        # reduced_fl = tf.reduce_max(fl, axis=1)
        reduced_fl = tf.reduce_sum(fl, axis=1, name='focal_loss')  # same as reduce_max
        return reduced_fl

    def hierarchical_attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        """
        Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
        The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
         for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
        Variables notation is also inherited from the article

        Args:
            inputs: The Attention inputs.
                Matches outputs of RNN/Bi-RNN layer (not final state):
                    In case of RNN, this must be RNN outputs `Tensor`:
                        If time_major == False (default), this must be a tensor of shape:
                            `[batch_size, max_time, cell.output_size]`.
                        If time_major == True, this must be a tensor of shape:
                            `[max_time, batch_size, cell.output_size]`.
                    In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                    the backward RNN outputs `Tensor`.
                        If time_major == False (default),
                            outputs_fw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[batch_size, max_time, cell_bw.output_size]`.
                        If time_major == True,
                            outputs_fw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_fw.output_size]`
                            and outputs_bw is a `Tensor` shaped:
                            `[max_time, batch_size, cell_bw.output_size]`.
            attention_size: Linear size of the Attention weights.
            time_major: The shape format of the `inputs` Tensors.
                If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
                If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
                Using `time_major = True` is a bit more efficient because it avoids
                transposes at the beginning and end of the RNN calculation.  However,
                most TensorFlow data is batch-major, so by default this function
                accepts input and emits output in batch-major form.
            return_alphas: Whether to return attention coefficients variable along with layer's output.
                Used for visualization purpose.
        Returns:
            The Attention output `Tensor`.
            In case of RNN, this will be a `Tensor` shaped:
                `[batch_size, cell.output_size]`.
            In case of Bidirectional RNN, this will be a `Tensor` shaped:
                `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
        """

        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)

        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        if tf.__version__ > '1.5.0':
            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
            alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

            # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
            output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
        else:
            inputs_shape = inputs.shape  # ( batch_size , seq_len, hidden_size)
            sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer

            # (batch_size * seq_len, hidden_size) * (hidden_size , attention_size) + （1，attention_size)
            #  = （batch_size * seq_len, attention_size)
            v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), w_omega) + tf.reshape(b_omega, [1, -1]))  # broadcasting
            # vu : (batch_size * seq_len, 1)
            vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
            # exps : (batch_size, seq_len)
            exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
            # Attention Vector
            alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

            # Output of Bi-RNN is reduced with attention vector
            # output = sum( alpha * h ) 加权之和
            output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)

        if not return_alphas:
            return output
        else:
            return output, alphas

    def bi_attention(self, H, hidden_size, max_len):
        """
        bi attention
        :param H:
        :param hidden_size:
        :param max_len:
        :return:
        """
        M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))
        alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1])))
        r = tf.matmul(tf.transpose(H, [0, 2, 1]),
                      tf.reshape(alpha, [-1, max_len, 1]))
        # r = tf.squeeze(r)
        r = tf.squeeze(r, axis=[2])
        h_star = tf.tanh(r)  # (batch , HIDDEN_SIZE

        return h_star
