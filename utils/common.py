import random

import numpy as np
import torch
import tensorflow as tf


def set_rand_seed(seed):
    """Sets random seed for reproducibility."""
    # 设置 Python 自身的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 TensorFlow 的随机种子
    tf.random.set_seed(seed)

    torch.manual_seed(seed)


def set_env_seed(seed):
    # 设置 Python 自身的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)