# -*- coding: utf-8 -*-

import logging
import os
import random
import sys
from logging.handlers import TimedRotatingFileHandler

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
file_handler = logging.FileHandler(save_dir+"test.log")
file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter  # 也可以直接给formatter赋值
# 为logger添加的日志处理器
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# 指定日志的最低输出级别，默认为WARN级别
logger.setLevel(logging.INFO)
# 输出不同级别的log
# logger.debug('this is debug info')
# logger.info('this is information')
# logger.warning('this is warning message')
# logger.error('this is error message')
# logger.critical('this is critical message')
#
# x = 1
# b = [i for i in range(0, 100)]
# while x < 2:
#     logger.info(len(b))
#     index = random.randint(0, len(b)-1)
#     # logging.warning("{}".format(index))
#     logger.warning("%s", index)           #或者这种写法
#     the_number = b.pop(index)
#     x += 1
#     logger.error(the_number)