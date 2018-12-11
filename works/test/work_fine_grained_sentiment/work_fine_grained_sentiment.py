#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/6/26'
# 
"""
import json
import os

import pandas as pd
import numpy as np
from tensorflow.contrib import learn
from sklearn.metrics import classification_report, f1_score

from tools.utility import Utility
from works.test.base.estimate_base import EstimateBase


class FineGrainedSentimentInfer(EstimateBase):
    def __init__(self):
        super().__init__()
        self.load_config('aspect_pro')

        self.reload_model()
        self.log = Utility.get_logger('aspect_pro')
        self.content_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(self.config['vocab_pkl'])

    def batch_iter_test(self, data, batch_size):
        """

        :param data:
        :param batch_size:
        :return:
        """
        assert isinstance(data, pd.core.frame.DataFrame), 'test data should be a DataFrame'
        content = data.content_token.values.tolist()
        data = np.array(list(self.content_vocab_processor.transform(content))).tolist()

        return super().batch_iter_test(data, batch_size)

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
            all_predictions.append(batch_predictions.T)
        all_predictions = np.concatenate(all_predictions)

        df['y'] = all_predictions.tolist()
        return df

    def run(self):
        """

        :return:
        """
        self.run_test()

    def test(self, df, out_path, with_pre=False):
        """

        :param df:
        :param out_path:
        :param with_pre:
        :return:
        """
        # df = df.head(10)
        if df is None or len(df) == 0:
            self.log.info('parse_dataset is empty')
            return

        if 'y' in df.columns:
            df.rename(columns={'y': 'y_'}, inplace=True)

        df = self.estimate(df)
        if df is None or len(df) == 0:
            self.log.info('estimate result is empty')
            return

        if 'y_' not in df.columns:
            # 无标签测试数据
            return self.test_no_label(df, out_path)

        def process(row):
            lab = eval(row['y_'])
            cl = []
            for i in range(0, 80, 4):
                lt = lab[i: i + 4]
                cl.append(np.argmax(lt))

            row['label'] = cl
            return row

        df = df.apply(process, axis=1)
        y_pre = df.y
        y_pre = np.array(y_pre.tolist())

        y_true = df.label
        y_true = np.array(y_true.tolist())

        f_scores = []
        for i in range(20):
            f = f1_score(y_true[:, i], y_pre[:, i], average='macro')
            f_scores.append(f)

        self.log.info('f1 score : {}'.format(f_scores))
        f_avg = np.array(f_scores).mean()
        self.log.info('mean f1 score: {}'.format(f_avg))

        df.to_csv(out_path, index=False, encoding='utf-8')

    def test_no_label(self, ret_df, out_path):
        """
        无标签数据测试
        :param ret_df:
        :return:
        """
        aspect = ['location_traffic_convenience',
                  'location_distance_from_business_district', 'location_easy_to_find',
                  'service_wait_time', 'service_waiters_attitude',
                  'service_parking_convenience', 'service_serving_speed', 'price_level',
                  'price_cost_effective', 'price_discount', 'environment_decoration',
                  'environment_noise', 'environment_space', 'environment_cleaness',
                  'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
                  'others_overall_experience', 'others_willing_to_consume_again']
        lab_dict = {
            0: 0,
            1: 1,
            2: -2,
            3: -1
        }

        df_ret = ret_df[['id', 'content', 'y']]

        def process(row):
            # y = eval(row['y'])
            y = row['y']
            for i, a in enumerate(y):
                row[aspect[i]] = lab_dict[a]
            return row

        df_ret = df_ret.apply(process, axis=1)

        df_ret = df_ret.drop(['y'], axis=1)
        df_ret.to_csv(out_path, index=False, encoding='utf-8')
