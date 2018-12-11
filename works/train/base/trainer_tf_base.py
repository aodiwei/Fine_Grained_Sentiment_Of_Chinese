#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/12/25'
# tensorflow 深度学习基类
"""
import os
from abc import abstractmethod
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import f1_score

from tools.utility import DfUtility

from works.train.base.trainer_base import TrainerBase


class TrainerTfBase(TrainerBase):
    def __init__(self):
        super().__init__()
        self.log = None
        self.max_ret = 0  # 记录最大的评价指标

    @abstractmethod
    def build_model(self):
        """
        子类实现
        :return: 
        """

    @abstractmethod
    def batch_iter_train(self, batch_size, num_epochs):
        """

        :param batch_size: 
        :param num_epochs: 
        :return: 
        """

    @abstractmethod
    def batch_iter_test(self, batch_size):
        """

        :param batch_size: 
        :return: 
        """

    def grads_and_vars(self, grads_and_vars):
        """
        对权重可视化稀疏情况
        :param grads_and_vars: 
        :return: 
        """
        for g, v in grads_and_vars:
            if g is not None:
                tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
        summary_op = tf.summary.merge_all()

        return summary_op

    def summary_writer(self, summary_name, is_train=True):
        """
        
        :param is_train: 
        :param summary_name: object
        :return: 
        """
        if self.model_out_path is None:
            raise ValueError('model_out_path is None')
        summ_root = os.path.join(self.model_out_path, 'summaries')
        if not os.path.exists(summ_root):
            os.mkdir(summ_root)
        summary_dir = os.path.join(summ_root, summary_name)
        if is_train:
            summary_writer = tf.summary.FileWriter(summary_dir, graph=tf.get_default_graph())
        else:
            summary_writer = tf.summary.FileWriter(summary_dir)

        return summary_writer

    def loss_acc_summary(self, loss, acc, f1=None):
        """
        创建loss acc的summary
        :param f1:
        :param loss:
        :param acc: 
        :return: 
        """
        summary = tf.Summary()
        summary.value.add(tag='loss', simple_value=loss)
        summary.value.add(tag='accuracy', simple_value=acc)
        if f1 is not None:
            summary.value.add(tag='f1', simple_value=f1)

        return summary

    def cal_f1_score(self, predictions, labels):
        """
        计算f1
        :param predictions:
        :param labels:
        :return:
        """

        def process(row):
            cl = []
            for i in range(0, 80, 4):
                x = int(i / 4)
                if self.train_args['special_class'] and x not in self.train_args['special_class']:
                    continue
                lt = row[i: i + 4]
                cl.append(np.argmax(lt))

            return cl

        y_pre = predictions
        y_true = np.array([process(x) for x in labels])

        f_scores = []
        for i in range(y_true.shape[1]):
            f = f1_score(y_true[:, i], y_pre[:, i], average='macro')
            f_scores.append(f)

        f_avg = np.array(f_scores).mean()
        f_avg = np.round([f_avg], decimals=4).tolist()[0]

        f_scores = np.round(f_scores, decimals=4).tolist()

        return f_scores, f_avg

    def train_step(self, x_batch, y_batch, epoch):
        """
        
        :param x_batch: 
        :param y_batch: 
        :param epoch: 
        :return: 
        """
        feed_dict = {
            self.model.input_x: x_batch,
            self.model.input_y: y_batch,
            self.model.dropout_keep_prob: self.train_args['dropout_keep_prob']
        }
        _, step, train_loss, train_accuracy = self.sess.run(
            [self.train_op, self.global_step, self.model.loss, self.model.accuracy],
            feed_dict)
        if step % self.train_args['log_every'] == 0:
            self.log.info('train: epoch: {}, step {}, loss {:g}, acc {:g}'.format(epoch, step, train_loss, train_accuracy))

        if step % self.train_args['evaluate_every'] == 0:
            summaries, predictions = self.sess.run([self.summary_op_all, self.model.predictions], feed_dict=feed_dict)
            f_scores, f_1 = self.cal_f1_score(predictions.T, y_batch)
            self.log.info('=====>train: epoch: {}, step {}, avg f1: {} \n all f1: {}'.format(epoch, step, f_1, f_scores))
            self.summary_train.add_summary(summaries, step)
            # 记录训练集 loss acc
            summary = self.loss_acc_summary(train_loss, train_accuracy, f_1)
            self.summary_train.add_summary(summary, step)

        return step

    def test_step(self, epoch, step):
        """
        
        :param step: 
        :param epoch: 
        :return: 
        """
        losses = []
        accuracies = []
        predictions = []
        y_batch_devs = []
        batches_dev = self.batch_iter_test(self.train_args['test_batch_size'])
        for x_batch_dev, y_batch_dev, _ in batches_dev:
            feed_dict = {
                self.model.input_x: x_batch_dev,
                self.model.input_y: y_batch_dev,
                self.model.dropout_keep_prob: 1
            }
            loss, accuracy, pre = self.sess.run([self.model.loss, self.model.accuracy, self.model.predictions], feed_dict)
            losses.append(loss)
            accuracies.append(accuracy)
            predictions.append(pre.T)
            y_batch_devs.append(y_batch_dev)

        predictions = np.concatenate(predictions)
        y_batch_devs = np.concatenate(y_batch_devs)
        f_scores, f_1 = self.cal_f1_score(predictions, y_batch_devs)
        self.log.info('=====>test: epoch: {}, step {}, avg f1: {} \n all f1: {}'.format(epoch, step, f_1, f_scores))

        saved_flag = False

        if self.max_ret == 0:
            self.saver_max_checkpoint_path = self.checkpoint_path.replace('checkpoints', 'checkpoints_max')
            if not os.path.exists(self.saver_max_checkpoint_path):
                os.mkdir(self.saver_max_checkpoint_path)
            self.saver_max_checkpoint_prefix = os.path.join(self.saver_max_checkpoint_path, "model")

        if f_1 > self.max_ret and f_1 > 0.688:  # 1026 0.6862:  # 1024
            self.saver_max.save(self.sess, self.saver_max_checkpoint_prefix, global_step=step)
            self.max_ret = f_1
            self.log.info(
                '=====>test: epoch: {}, step {}, saved max f_1[{}] model to {} '.format(epoch, step, f_1, self.saver_max_checkpoint_prefix))
            saved_flag = True

        # 计算整个测试集的平均值
        test_loss_avg = np.mean(losses)
        test_accuracy_avg = np.mean(accuracies)
        # 记录测试集loss acc
        summary = self.loss_acc_summary(test_loss_avg, test_accuracy_avg, f_1)
        self.summary_test.add_summary(summary, step)
        self.log.info('test: epoch: {}, step {}, loss {:g}, acc {:g}'.format(epoch, step, test_loss_avg, test_accuracy_avg))

        return saved_flag

    def run(self):
        """
        training
        :return: 
        """
        self.log.info('****************************starting training***********************************')
        re_train = self.re_model_path is not None
        if not re_train:
            self.create_timestamp_folder()
        elif not os.path.exists(self.re_model_path):
            raise EOFError('retrain model {} is not exists'.format(self.re_model_path))
        else:
            self.model_out_path = self.re_model_path

        # Training
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=self.train_args['allow_soft_placement'],
                                          log_device_placement=self.train_args['log_device_placement'], )
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                #  build 模型
                self.build_model()

                if self.train_args['pre_word']:
                    words_embedd_vect = self.load_pre_word_embedding()
                    self.sess.run(self.model.embedding_init, feed_dict={self.model.embedding_placeholder: words_embedd_vect})

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                # self.lr = tf.train.exponential_decay(learning_rate=self.train_args['learn_rate'], global_step=self.global_step,
                                                     # decay_steps=self.train_args['lr_decay_steps'],
                                                     # decay_rate=0.96, staircase=True, name='learn_rate')
                # # optimizer = tf.train.AdamOptimizer(self.lr)
                optimizer = tf.train.AdamOptimizer(self.train_args['learn_rate'])
                # optimizer = tf.train.GradientDescentOptimizer(self.train_args['learn_rate'])
                # 以下两步相当于 optimizer.minimize() 但是拿到了grads_and_vars，用于可视化
                grads_and_vars = optimizer.compute_gradients(self.model.loss)
                self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step, name='train_op')

                self.summary_op_all = self.grads_and_vars(grads_and_vars)
                self.summary_train = self.summary_writer(summary_name='train')
                self.summary_test = self.summary_writer(summary_name='test', is_train=False)

                # 这一句一定要放在重载模型前面
                self.sess.run(tf.global_variables_initializer())

                saver = tf.train.Saver(save_relative_paths=True, max_to_keep=1)
                self.saver_max = tf.train.Saver(save_relative_paths=True, max_to_keep=1)
                if re_train:
                    re_model = os.path.join(self.config.get('re_model_path'), 'checkpoints')
                    self.checkpoint_path = re_model
                    self.log.info('reload model from {}'.format(re_model))
                    checkpoint_file = tf.train.latest_checkpoint(re_model)
                    self.checkpoint_prefix = os.path.join(re_model, 'model')
                    saver.restore(self.sess, checkpoint_file)

                batches = self.batch_iter_train(self.train_args['batch_size'], self.train_args['num_epochs'])
                for x_batch, y_batch, epoch in batches:
                    try:
                        # 训练
                        step = self.train_step(x_batch, y_batch, epoch)

                        # 评估测试集
                        saved_flag = False
                        if step % self.train_args['evaluate_every'] == 0:
                            saved_flag = self.test_step(epoch, step)

                        # 保存模型
                        if not saved_flag and step % self.train_args['checkpoint_every'] == 0:
                            path = saver.save(self.sess, self.checkpoint_prefix, global_step=step)
                            self.log.info('last saved model checkpoint to {} at step {}, epoch {}'.format(path, step, epoch))

                    except KeyboardInterrupt as e:
                        path = saver.save(self.sess, self.checkpoint_prefix, global_step=step)
                        self.log.info('saved KeyboardInterrupt model checkpoint to {} at step {}, epoch {}'.format(path, step, epoch))
                        raise e
                path = saver.save(self.sess, self.checkpoint_prefix, global_step=step)
                self.summary_train.close()
                self.summary_test.close()
                self.log.info('last saved model checkpoint to {} at step {}'.format(path, step))
                self.log.info('****************************finish training***********************************')

    def batch_data(self, batch_size, num_epochs, df_path, x_col, y_col, class_num=2, onehot_y=True, fun=None, enhance=False):
        """
        df数据分割为batch， iterator: 是否用chunk的方式fetch csv数据，如果文件较大推荐使用此方式
        :param enhance:
        :type onehot_y: object
        :param fun: 对x的特殊处理
        :param class_num: 
        :param batch_size: 
        :param num_epochs: 
        :param df_path: 
        :param x_col: x 列名list, 如果只有一个列，不用list
        :param y_col: 
        :return: 
        """
        # assert isinstance(x_col, list), 'x_col must be list'
        assert isinstance(y_col, str), 'y_col must be str'
        assert os.path.exists(df_path), '{} not exists'.format(df_path)

        iterator = self.config.get('csv_iterator', False)
        if not isinstance(x_col, list):
            usecols = [x_col] + [y_col]
        else:
            usecols = x_col + [y_col]
        df = pd.read_csv(df_path, iterator=iterator, usecols=usecols, lineterminator='\n')
        step = 0
        for epoch in range(1, num_epochs + 1):
            if not iterator:
                df = df.sample(frac=1)

            start = 0
            end = batch_size
            while 1:
                if iterator:
                    try:
                        data = df.get_chunk(batch_size)
                        data = data.sample(frac=1)
                    except StopIteration as e:
                        # shuffle_data(path)
                        df = pd.read_csv(df_path, iterator=iterator, usecols=x_col + [y_col])
                        break
                else:
                    data = df[start:end]
                    if len(data) == 0:
                        break
                    data = data.sample(frac=1)
                    start += batch_size
                    end += batch_size
                step += 1
                x = data[x_col].values
                labels = data[y_col].values
                if fun is not None:
                    x = fun(x, step, epoch)
                # 0: [1, 0] 为正常， 1: [0, 1]为恶意
                if onehot_y:
                    y = DfUtility.encode_one_hot(class_num, labels)
                else:
                    y = []
                    for l in labels:
                        y.append(eval(l))
                    # y = labels

                yield x, y, epoch

    def vis_embeddings(self):
        """

        :return:
        """
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = self.model.embedding_W.name
        embedding.metadata_path = os.path.join(self.model_out_path, 'metadata.tsv')
        summary_writer = tf.summary.FileWriter(self.model_out_path)
        # 保存embedding
        projector.visualize_embeddings(summary_writer, config)

    def load_pre_word_embedding(self):
        """
        加载预训练词向量
        :return:
        """
        embedd_path = self.config.get('pre_word_vect')
        with open(embedd_path, 'r') as f:
            words_embedd_vect = json.load(fp=f)
        self.log.info('load pre trained word: {}'.format(embedd_path))
        return words_embedd_vect
