#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2017/5/19'
#
"""
import datetime
import logging
import os
import pathlib
import re
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count, dummy, Process
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import yaml

import define
from tools.const import Const


class Utility:
    """
    通用的与业务相关的工具
    """
    __logger = {}

    __config = {}

    __subnet_range = []

    __syslog = None

    alphabet_dict = None

    @classmethod
    def get_conf(cls, config_name='root'):
        """
        config
        :param config_name: 以dev结尾、以pro结尾、
        :return: conf
        """
        if config_name in cls.__config:
            return cls.__config[config_name]

        if config_name.endswith('_dev'):  # 开发环境配置
            path = os.path.join(define.root, 'works', 'develop', 'config_dev', '{}.yaml'.format(config_name))
        elif config_name.endswith('_pro'):  # 生产环境配置
            path = os.path.join(define.root, 'works', 'product', 'config_pro', '{}.yaml'.format(config_name))
        else:  # 公用环境配置
            path = os.path.join(define.root, 'config.yaml')
        if not os.path.exists(path):
            raise FileNotFoundError('{}'.format(path))

        with open(path, 'r', encoding='utf-8') as s:
            config = yaml.load(s)

        cls.__config[config_name] = config

        return cls.__config[config_name]

    @classmethod
    def fetch_config_root_child(cls, config_name, section):
        """
        如果不是root，优先从子配置获取，如果没有则从root获取
        :param config_name: 
        :param section: 
        :return: 
        """
        config = cls.get_conf(config_name)
        if config_name == 'root':
            val = config.get(section)
        else:
            val = config.get(section, None)
            if val is None:
                config_root = cls.get_conf('root')
                val = config_root.get(section)

        return val

    @classmethod
    def get_logger(cls, log_name=None):

        conf = cls.get_conf().get('logger_access')
        # assert conf is not None, "no {} in config file".format(log_name)
        log_path = conf.get("path")
        level = conf.get("level")
        is_console = conf.get("is_console")
        if log_name is None:
            log_name = 'root'
        logger = logging.getLogger(log_name)

        logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        if log_path:
            file_handler = logging.FileHandler(os.path.join(log_path, '{}.log'.format(log_name)), encoding="UTF-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if is_console:
            stream_handler = logging.StreamHandler(sys.stderr)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        return logger

    @classmethod
    def strptime(cls, datetime_str, inst=True):
        """
        format str to datetime instance,
        format including:
            YYYY-MM-DD
            YYYY/MM/DD
            YYYY_MM_DD
            YYYY-MM-DD %H:%M:%S
            YYYY/MM/DD %H:%M:%S
            YYYY_MM_DD %H:%M:%S
            YYYY-M-D
            YYYY/M/D
            YYYY_M_D
            YYYY-M-D %H:%M:%S
            YYYY/M/D %H:%M:%S
            YYYY_M_D %H:%M:%S
        :param inst:
        :param datetime_str:
        :return:datetime instance/string format
        """
        p_date = re.compile(r"^(\d{4})[-/_](\d{1,2})[-/_](\d{1,2})$")
        p_datetime = re.compile(r"^(\d{4})[-/_](\d{1,2})[-/_](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})$")
        result_date = p_date.findall(datetime_str)
        result_datetime = p_datetime.findall(datetime_str)
        if result_datetime:
            result = result_datetime[0]
            # this way perform better 7x than strptime
            datetime_inst = datetime.datetime(int(result[0]), int(result[1]), int(result[2]), int(result[3]),
                                              int(result[4]), int(result[5]), int(result[0]))
            datetime_str = datetime_inst.strftime(Const.LOCAL_FORMAT_DATETIME)
        elif result_date:
            result = result_date[0]
            datetime_inst = datetime.datetime(int(result[0]), int(result[1]), int(result[2]))
            datetime_str = datetime_inst.strftime(Const.LOCAL_FORMAT_DATE)
        else:
            raise ValueError("nonsupport datetime format: {}".format(datetime_str))

        return datetime_inst if inst else datetime_str

    @classmethod
    def timestamp2str(cls, timestamp=None):
        """
        时间戳转为时间字符串
        :param timestamp: 10位数的int
        :return: 
        """
        if timestamp is None:
            timestamp = time.time()
        return time.strftime(Const.LOCAL_FORMAT_DATETIME, time.localtime(timestamp))

    @classmethod
    def timestamp2str_file(cls, timestamp=None):
        """
        时间戳转为时间字符串文件名
        :param timestamp: 10位数的int
        :return: 
        """
        if timestamp is None:
            timestamp = time.time()
        return time.strftime(Const.LOCAL_FORMAT_DATETIME_FILE, time.localtime(timestamp))

    @classmethod
    def pp_info(cls, obj):
        """
        pprint 格式化打印
        :param obj: 
        :return: 
        """
        import pprint
        log = cls.get_logger()
        log.info(pprint.pformat(obj))

    @classmethod
    def multiprocess_map_async(cls, fun, iterator, rate=0.75):
        """
        多进程-异步
        :param rate: 使用率
        :param fun: 
        :param iterator: 
        :return: 
        """
        cpu = os.cpu_count()
        process_count = int(cpu * rate)  # 使用75%的CPU资源
        if '__len__' in iterator.__dir__():
            iter_len = len(iterator)
        elif 'gi_code' in iterator.__dir__():
            iter_len = iterator.gi_code.co_consts[1]
        else:
            iter_len = process_count

        process_count = process_count if iter_len > process_count else iter_len

        pool = Pool(process_count)
        ret = pool.map_async(fun, iterator)
        pool.close()
        pool.join()

        return ret.get()

    @classmethod
    def multiprocess_async(cls, fun, iterator, *args):
        """
        多进程-异步 ret = utility.Utility.multiprocess_async(split_, gs, *(df, th))
        :param fun: 
        :param iterator: 
        :return: 
        """
        cpu = os.cpu_count()
        process_count = int(cpu * 0.75)  # 使用75%的CPU资源
        if '__len__' in iterator.__dir__():
            iter_len = len(iterator)
        elif 'gi_code' in iterator.__dir__():
            iter_len = iterator.gi_code.co_consts[1]
        else:
            iter_len = process_count

        process_count = process_count if iter_len > process_count else iter_len

        pool = Pool(process_count)

        results = []
        for item in iterator:
            args_ = (item,) + args
            result = pool.apply_async(fun, args=args_)
            results.append(result)

        pool.close()
        pool.join()

        results = [result.get() for result in results]

        return results

    @classmethod
    def multiprocess_task(cls, targets, kwargs_list=None):
        """
        多进程，使用于不同target的情况
        :param targets: fun
        :param kwargs_list: kwargs_list = [None, None, {'i': 555555}]
        :return:
        """
        works = []
        for i, t in enumerate(targets):
            if kwargs_list is not None:
                kwargs = kwargs_list[i] if kwargs_list[i] is not None else {}
            else:
                kwargs = {}
            work = Process(target=t, kwargs=kwargs)
            works.append(work)

        for work in works:
            work.start()

        for work in works:
            work.join()

    @classmethod
    def multithreading_map_async(cls, fun, iterator, rate=0.75):
        """
        多线程-异步
        :param rate: 使用率
        :param fun:
        :param iterator:
        :return:
        """
        cpu = os.cpu_count()
        process_count = int(cpu * rate)
        if '__len__' in iterator.__dir__():
            iter_len = len(iterator)
        elif 'gi_code' in iterator.__dir__():
            iter_len = iterator.gi_code.co_consts[1]
        else:
            iter_len = process_count

        process_count = process_count if iter_len > process_count else iter_len

        pool = dummy.Pool(process_count)
        ret = pool.map_async(fun, iterator)
        pool.close()
        pool.join()

        return ret.get()

    @classmethod
    def multithreading_task(cls, targets, kwargs_list=None):
        """
        多线程，使用于不同target的情况
        :param targets: fun
        :param kwargs_list: kwargs_list = [None, None, {'i': 555555}]
        :return:
        """
        works = []
        for i, t in enumerate(targets):
            if kwargs_list is not None:
                kwargs = kwargs_list[i]
            else:
                kwargs = {}
            work = dummy.Process(target=t, kwargs=kwargs)
            works.append(work)

        for work in works:
            work.start()

        for work in works:
            work.join()

    @classmethod
    def coroutine_task(cls, targets, kwargs_list=None):
        """
        多线程，使用于不同target的情况
        :param targets: fun
        :param kwargs_list: kwargs_list = [None, None, {'i': 555555}]
        :return:
        """
        import gevent
        from gevent import monkey
        monkey.patch_all()
        works = []
        for i, t in enumerate(targets):
            if kwargs_list is not None:
                kwargs = kwargs_list[i]
            else:
                kwargs = {}
            work = gevent.spawn(t, **kwargs)
            works.append(work)

        ret = gevent.joinall(works)

        ret = [x.value for x in ret]

        return ret

    @classmethod
    def coroutine_fun(cls, fun, iterator):
        """
        协程处理
        :param fun: 
        :param iterator: 
        :return: 
        """
        import gevent
        from gevent import monkey
        monkey.patch_all()
        tasks = [gevent.spawn(fun, i) for i in iterator]
        ret = gevent.joinall(tasks)
        ret = [x.value for x in ret]

        return ret

    @classmethod
    def init_alphabet_dict(cls):
        """
        
        :return: 
        """
        if cls.alphabet_dict is None:
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-._'
            cls.alphabet_dict = dict(zip(alphabet, range(1, len(alphabet) + 1)))
        return cls.alphabet_dict

    @classmethod
    def bak_data(cls, src):
        """

        :param src:
        :param dst:
        :return:
        """
        bad_path = r'/home/bak_data'
        if not os.path.exists(bad_path):
            os.mkdir(bad_path)
        filename = os.path.split(src)[-1]
        filename = os.path.join(bad_path, filename)
        shutil.copy(src, filename)

    @classmethod
    def walk_path(cls, root_dir, files=None):
        files = [] if files is None else files
        for lists in os.listdir(root_dir):
            path = os.path.join(root_dir, lists)
            if os.path.isdir(path):
                cls.walk_path(path, files)
            else:
                # print(path)
                files.append(path)
        return files

    @classmethod
    def before_runner(cls):
        """
        当用cx_Freeze 打包后在由于未知原因tensorflow模块不能正常使用，需要包tf拷贝到系统库目录
        :return:
        """
        if not os.path.exists('lib'):
            print('developing env')
            return None
        else:
            print('product env')
        p = r'/usr/local/python3/lib/python3.5/site-packages/tensorflow'
        if not os.path.exists(p):
            pathlib.Path(os.path.split(p)[0]).mkdir(parents=True, exist_ok=True)
            os.system('cp -r lib/tensorflow {}'.format(p))
            print('copy lib/tensorflow to {}'.format(p))


class DfUtility:
    """
    数据相关的通用工具，操作对象主要为df
    """

    @classmethod
    def get_top_N(cls, df, columns, n=None, limit=None):
        """
        获取前M个分组的所有数据
        :param limit: 如果有大小限制
        :param df: 
        :param n: 
        :param columns: 
        :return: list[dataframe]
        """
        groups = df.groupby(columns)
        # 找出top N的分组的组名
        if n is None:
            top = groups.size().sort_values(ascending=False)
        else:
            top = groups.size().sort_values(ascending=False).nlargest(n)
        top_g = top.index.get_values()

        # 获取top N的分组
        if limit is None:
            filter_group = [groups.get_group(item).reset_index(drop=True) for item in top_g]
        else:
            filter_group = [groups.get_group(item).reset_index(drop=True) for item in top_g if len(groups.get_group(item)) >= limit]

        return filter_group

    @classmethod
    def drop_max(cls, df, col, n=1):
        """
        去每列除最大值
        :param df: 
        :param col: 操作的列
        :param n: top n
        :return: 
        """
        count = len(df)
        for _ in range(n):
            idx = df[col].idxmax()
            df = df.drop(idx)
        print('drop count: {}'.format(count - len(df)))
        return df

    @classmethod
    def drop_min(cls, df, col, n=1):
        """
        去除最大值
        :param df: 
        :param col: 操作的列
        :param n: top n
        :return: 
        """
        count = len(df)
        for _ in range(n):
            idx = df[col].idxmin()
            df = df.drop(idx)
        print('drop count: {}'.format(count - len(df)))
        return df

    @classmethod
    def split_df_data_by_cpu_count(cls, df, rate=0.75):
        """
        根据cpu个数分割数据，用于多进程计算
        :param rate: 利用率
        :param df: 
        :return: 
        """
        cpu_num = int(cpu_count() * rate)
        dfs = cls.split_df(df, count=cpu_num)

        return dfs

    @classmethod
    def split_df(cls, df, gap=None, count=None):
        """
        切分df为多个df
        :param count: 
        :param df: 
        :param gap:  per_df_count
        :return: 
        """
        # assert (gap and count) is None and (gap or count) is not None, 'just one of gap and count should be None'
        ass = lambda x: x is None
        assert ass(gap) ^ ass(count), 'just one of gap and count should be None'

        data_num = len(df)
        if gap is None:
            gap = data_num / count
            gap = int(pd.np.ceil(gap))
        if count is None:
            count = data_num / gap
            count = int(pd.np.ceil(count))

        dfs = [df[x * gap:x * gap + gap] for x in range(count)]
        dfs = [x for x in dfs if not x.empty]

        return dfs

    @classmethod
    def difference_set(cls, df_s, df_d):
        """
        差集，从df_s中去除df_d
        :param df_s: 
        :param df_d: 
        :return: 
        """
        merged = df_s.merge(df_d, indicator=True, how='outer')
        df_temp = merged[merged['_merge'] == 'left_only']
        df_temp.drop('_merge', axis=1, inplace=True)

        return df_temp

    @classmethod
    def train_test_split(cls, df, test_size):
        """
        df数据分割为训练和测试
        :param df: 
        :param test_size: 百分比
        :return: 
        """
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=test_size)

        # msk = pd.np.random.rand(len(df)) < test_size
        # test = df[msk]
        # train = df[~msk]

        return train, test

    @classmethod
    def encode_one_hot(cls, class_num, label):
        """
        one - hot 
        对角矩阵的方式实现，不论类别出现的顺序怎样都会得到一样的结果，
        用pd的get_dummies会有类别出现顺序不一样得到的结果也不一样        
        :param class_num: 
        :param label: 为数字list，比如 [0,1,2]
        :return: np.array
        """
        met = np.eye(class_num)
        return met[label]

    @classmethod
    def urlparse_domain(cls, row):
        """
        url提取domain
        :param row: 
        :return: 
        """
        ulr_obj = urlparse(row['url'])
        netloc = ulr_obj.netloc
        if netloc.startswith('www.'):
            netloc = netloc.lstrip('www.')
        row['domain'] = netloc
        return row

    @classmethod
    def clean(cls, row):
        """
        清理文本中的各种符号，繁转简，分词（中英文）
        :param row:
        :return:
        """
        import jieba
        from hanziconv import HanziConv
        title = row.title.strip().lower()
        title = re.sub(r'[\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007F\u2000-\u206f\u2500-\u257F'
                       r'\u3000-\u303f\uff00-\uffef\u00A0-\u00BF\uFFF0-\uFFFF\u30fb\u2605♪™丨]', '', title)
        #     print('1.====>', title)
        if re.findall(r'[\u4E00-\u9FA5]', title):
            title = HanziConv.toSimplified(title)
            #         print('2.====>', title)
        if re.findall(r'[\u2E80-\u9FFF]', title):
            title_cut = jieba.cut(title)
            title_cut = [x for x in title_cut if x != '' and x != ' ' and x != '\t' and x != '\n']
            title = ' '.join(title_cut)
            print('3.====>', title)
        else:
            title = title.split()
            title = [x for x in title if x != '' and x != ' ' and x != '\t' and x != '\n']
            title = ' '.join(title)
            print('4.====>', title)
        row['title'] = title
        return row

    @classmethod
    def parse_decode_error_csv(cls, path, fields_index=None, fields=None, dtypes=None, replace_file=None):
        """
        pandas 读取csv由于编码异常导致失败的情况下的处理方式，ignore的方式打开文件，再重新写入文件
        :param path:
        :param fields_index:
        :param fields:
        :param dtypes:
        :param replace_file:
        :return:
        """
        df = None
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                content = f.read()
                with open(path, 'w', encoding='utf-8') as ff:
                    ff.write(content)
            df = pd.read_csv(path,
                             # quoting=csv.QUOTE_NONE,
                             error_bad_lines=False,
                             names=fields,
                             usecols=fields_index,
                             keep_default_na=False,
                             na_values=''
                             )
            df.dropna(inplace=True)
            if dtypes is not None:
                df = df.astype(dtypes)

            if replace_file is not None:
                df.to_csv(path.replace('.csv', '_r.csv'), index=False, encoding='utf-8')
        except ValueError as e:
            if df is not None:
                df = cls.clean_type_error(df, dtypes)
            else:
                print('fun parse_decode_error_csv ValueError and df is None: {}'.format(e))
        except Exception as e:
            print('fun parse_decode_error_csv: {}'.format(e))
            df = pd.DataFrame()

        return df

    @classmethod
    def strip_df_object(cls, df, dtypes, func=None):
        """
        去除字符串两边空格
        :param func: 自定义处理方式
        :param df:
        :param dtypes:
        :return:
        """
        if len(df) == 0:
            return df

        if func is None:
            func = lambda x: x.str.strip('"')
        for k, v in dtypes.items():
            if v == np.object:
                try:
                    df[k] = func(df[k])
                except Exception as e:
                    print(e)

        return df

    @classmethod
    def clean_type_error(cls, df, dtypes):
        """
        在astype时有异常的object数据在数值型数据里，需要剔除
        :param df:
        :param dtypes:
        :return:
        """
        try:
            dt = df.dtypes
            for k, v in dtypes.items():
                t = dt.get(k, None)
                if t is not None and t == np.object and v != np.object:
                    df_tmep = df[df[k].apply(lambda x: not str(x).isnumeric())]
                    if df_tmep.shape[0] > 0:
                        print('fun clean_type_error error type', df_tmep.shape[0])
                        df.drop(df_tmep.index, inplace=True)

            df = df.astype(dtypes)
        except Exception as e:
            print('fun clean_type_error: {}'.format(e))
            df = pd.DataFrame()

        return df
