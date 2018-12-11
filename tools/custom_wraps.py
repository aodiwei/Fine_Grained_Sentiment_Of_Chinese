#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = '修饰器'
__author__ = 'David Ao'
__mtime__ = '2017/5/22'
# NOTE TAHT 这个模块不要引用其他高级模块，否则会造成相互引用
"""
from functools import wraps
import time


def singleton(cls):
    """
    单例修饰器
    :param cls:
    :return:
    """
    instances = {}

    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper


def while_fun(func):
    """
    循环执行
    :return:
    """

    def wrapper(*args, **kwargs):
        sleep_time = kwargs.get('sleep_time', 0)
        loop_count = kwargs.get('loop_count', 0)
        # 只执行一次
        if sleep_time == 0 and loop_count == 0:
            func(*args, **kwargs)
        # 无限循环
        elif loop_count == 0:
            while 1:
                func(*args, **kwargs)
                time.sleep(sleep_time)
        # 有限循环
        else:
            for _ in loop_count:
                func(*args, **kwargs)
                time.sleep(sleep_time)

    return wrapper


def while_func(sleep_time=0):
    """
    无限循环
    :param sleep_time: 
    :return: 
    """

    def wrapper_decorator(func):
        def wrapper(*args, **kwargs):
            while 1:
                func(*args, **kwargs)
                time.sleep(sleep_time)

        return wrapper

    return wrapper_decorator


def exception_to_list(log=None):
    """
    异常时返回空list
    :param log: 
    :return: 
    """

    def wrapper_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                fun_name = func.__module__
                if log:
                    log.exception('{}: {}'.format(fun_name, e))
                else:
                    print('process exception return list, {}: {}'.format(fun_name, e))
                return []

        return wrapper

    return wrapper_decorator


def exception_to_None(func):
    """
    异常时返回None
    :return: 
    """
    fun_exception_counter = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            fun_name = func.__module__
            return func(*args, **kwargs)
        except Exception as e:
            print('process exception return None, exception times: {}'.format(e))
            # 记录异常次数，达到3次就抛出异常
            if fun_name not in fun_exception_counter:
                fun_exception_counter[fun_name] = 1
            else:
                fun_exception_counter[fun_name] += 1
            if fun_exception_counter[fun_name] > 3:
                print('exception times > 3 raise e: {}'.format(e))
                raise e
            time.sleep(60)
            return None

    return wrapper


def log_spend_time(log=None):
    """
    打印耗时
    :param log:
    :return:
    """

    def wrapper_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            ret = func(*args, **kwargs)
            t2 = time.time()

            fun_name = func.__module__
            msg = '{} spend {} second'.format(fun_name, t2 - t1)
            if log:
                log.debug(msg)
            else:
                print(msg)
            return ret

        return wrapper

    return wrapper_decorator
