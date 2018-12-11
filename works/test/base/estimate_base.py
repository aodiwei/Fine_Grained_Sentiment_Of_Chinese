#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/12/27'
# 
"""
import json
import os
import pandas as pd
from abc import ABCMeta, abstractmethod

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

from tools.utility import Utility


class EstimateBase(metaclass=ABCMeta):
    def __init__(self):
        self.log = None

    def load_config(self, config_name):
        """

        :param config_name: object
        :return: 
        """
        self.config = Utility.get_conf(config_name)
        self.re_model_path = self.config.get('re_model_path')
        self.test_batch_size = self.config.get('test_batch_size')
        self.send_syslog = self.config.get('send_syslog')
        self.save_db = self.config.get('save_db')
        self.save_file = self.config.get('save_file')
        self.process_inter = self.config.get('process_inter')
        self.db_fields = self.config.get('db_fields')
        self.test_path = self.config.get('test_path', None)

    def reload_model(self):
        """
        重载模型
        :return: 
        """
        max_path = os.path.join(self.re_model_path, 'checkpoints_max', 'checkpoint')
        if os.path.exists(max_path):
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(self.re_model_path, 'checkpoints_max'))
        else:
            checkpoint_file = tf.train.latest_checkpoint(os.path.join(self.re_model_path, 'checkpoints'))
        print('checkpoint_file {}'.format(checkpoint_file))
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file), clear_devices=True)
                saver.restore(self.sess, checkpoint_file)

                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.predictions = graph.get_operation_by_name("predictions").outputs[0]

    def estimate(self, df):
        """
        推理
        :param df: 
        :return: 
        """
        batches = self.batch_iter_test(df, self.test_batch_size)
        # 判断
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = self.sess.run(self.predictions, {self.input_x: x_test_batch, self.dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        df['y'] = all_predictions
        return df

    @abstractmethod
    def batch_iter_test(self, data, batch_size):
        """
        
        :param batch_size: 
        :param data: 
        :return: 
        """
        assert isinstance(data, list), 'test data should be a list'
        count = len(data)
        for i in range(0, count, batch_size):
            x = data[i: i + batch_size]
            yield x

    # @abstractmethod
    def test(self, df, out_path, with_pre=False):
        """

        :param df:
        :param out_path:
        :param with_pre:
        :return:
        """
        pass
        # if df is None or len(df) == 0:
        #     self.log.info('parse_dataset is empty')
        #     return
        # if 'y' in df.columns:
        #     df.rename(columns={'y': 'y_'}, inplace=True)
        #
        # df = self.estimate(df)
        # if df is None or len(df) == 0:
        #     self.log.info('estimate result is empty')
        #     return
        #
        # df.to_csv(out_path, index=False, encoding='utf-8')
        # self.log.info('{}=>0: {}, 1: {}'.format(out_path, len(df.query('y==0')), len(df.query('y==0'))))
        # if 'y_' not in df.columns:
        #     return
        # y = df.y.values
        # y_ = df.y_.values
        #
        # df_error_01 = df.query('y_==0 and y==1')
        # df_error_10 = df.query('y_==1 and y==0')
        # df_error_01.to_csv(out_path.replace('.csv', '_err01.csv'), index=False, encoding='utf-8')
        # df_error_10.to_csv(out_path.replace('.csv', '_err10.csv'), index=False, encoding='utf-8')
        #
        # self.pro_confusion_matrix(y_, y, out_path.replace('.csv', '_conf.png'))
        # rep = classification_report(y_, y)
        # with open(out_path.replace('.csv', '_report.json'), 'w') as f:
        #     json.dump(rep, f)
        #     self.log.info('{}: {}'.format(out_path, rep))

    def pro_confusion_matrix(self, y_test, y_pred, imagename):
        """

        :param y_test:
        :param y_pred:
        :param imagename:
        :return:
        """
        import matplotlib as mpl

        if os.environ.get('DISPLAY', '') == '':
            print('no display found. Using non-interactive Agg backend')
            mpl.use('Agg')
        import matplotlib.pyplot as plt

        con = confusion_matrix(y_test, y_pred)
        self.log.info(con)

        con_list = con.tolist()
        with open(imagename.replace('_conf.png', '_confusion.json'), 'w') as f:
            json.dump(con_list, f)
        plt.rcParams['figure.figsize'] = (40, 12)
        plt.matshow(con)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('Churned')
        plt.xlabel('Predicted')
        plt.savefig(imagename, dpi=300)

        # plt.show()

    def run_test(self):
        """
        如果数据不是csv，可以重载这个函数，转为df
        :return:
        """
        if self.test_path is None:
            self.log.warning('test path is empty, set path in .yaml file firstly')
            return

        model_name = os.path.split(self.re_model_path)[-1]
        curr = Utility.timestamp2str_file()
        out_folder = model_name + curr
        if os.path.isdir(self.test_path):
            out_folder = os.path.join(self.test_path, out_folder)
            os.mkdir(out_folder)
            fs = os.listdir(self.test_path)
            for f in fs:
                out_path = os.path.join(out_folder, f)
                f = os.path.join(self.test_path, f)
                if os.path.isdir(f):
                    continue
                self.log.info('input: {}'.format(f))
                self.log.info('output: {}'.format(out_path))
                df = pd.read_csv(f)
                self.test(df, out_path)
        elif os.path.isfile(self.test_path):
            fs = os.path.split(self.test_path)
            out_folder = os.path.join(fs[0], out_folder)
            os.mkdir(out_folder)
            out_path = os.path.join(out_folder, fs[-1])
            self.log.info('input: {}'.format(self.test_path))
            self.log.info('output: {}'.format(out_path))
            df = pd.read_csv(self.test_path, lineterminator='\n')
            self.test(df, out_path)
