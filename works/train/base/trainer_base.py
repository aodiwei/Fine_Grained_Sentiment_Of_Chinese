#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/12/25'
# 
"""
import os
from abc import abstractmethod, ABCMeta

from tools.utility import Utility


class TrainerBase(metaclass=ABCMeta):
    def __init__(self):
        self.config = None
        self.train_data_path = None
        self.test_data_path = None
        self.model_out_path = None
        self.checkpoint_path = None
        self.re_model_path = None
        self.train_args = None
        self.model = None

    @abstractmethod
    def build_model(self):
        """
        创建模型
        :return: 
        """

    @abstractmethod
    def run(self):
        """
        训练、继续训练
        :return: 
        """

    def load_config(self, config_name):
        """
        
        :param config_name: object
        :return: 
        """
        self.config = Utility.get_conf(config_name)
        self.train_data_path = self.config.get('train_data_path')
        self.test_data_path = self.config.get('test_data_path')
        self.model_out_path = self.config.get('model_out_path')
        self.re_model_path = self.config.get('re_model_path')
        self.train_args = self.config.get('train_args')
        print('train_args:==>\n',  self.train_args)

    def create_timestamp_folder(self, child='checkpoints'):
        """
        创建时间戳的文件夹，存放输出模型
        :param child: 
        :return: 
        """
        timestamp = Utility.timestamp2str_file()
        self.model_out_path = os.path.join(self.model_out_path, timestamp)
        if child is not None:
            self.checkpoint_path = os.path.join(self.model_out_path, child)
        else:
            self.checkpoint_path = self.model_out_path

        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.checkpoint_prefix = os.path.join(self.checkpoint_path, 'model')
