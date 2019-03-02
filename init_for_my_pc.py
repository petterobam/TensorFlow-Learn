# -*- coding: utf-8 -*-
# 全局取消证书验证 https 下载问题
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# 解决 Mac CPU 不支持问题 TensorFlow binary was not compiled to use: AVX2 FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 忽略 warning 控制台输出
import warnings
warnings.filterwarnings("ignore")