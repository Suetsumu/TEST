# -*- coding: utf-8 -*-

import logging
import os
import random
import sys
from logging.handlers import TimedRotatingFileHandler
class logConfig:
    def __init__(self,file_dir="test.log"):
        # 配置log的文件夹
        save_dir = './logs/'
        if not os.path.exists(save_dir) :
            os.makedirs(save_dir)
        # 获取logger实例，如果参数为空则返回root logger
        logger = logging.getLogger("MG-WFBP")
        # 指定logger输出格式
        format_str = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        formatter = logging.Formatter(format_str, datefmt)
        # 文件日志
        file_handler = logging.FileHandler(save_dir+file_dir)
        file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
        # 控制台日志
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = formatter  # 也可以直接给formatter赋值
        # 为logger添加的日志处理器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        # 指定日志的最低输出级别，默认为WARN级别
        logger.setLevel(logging.INFO)
        self.logger = logger
    def __getLogger__(self):
        return self.logger
