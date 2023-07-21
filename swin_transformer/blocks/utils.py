import tensorflow as tf 
from tensorflow import numpy 
import numpy as np 


def window_reverse(windows, window_size, h, w, c):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
        c (int): Number of channels in image

    Returns:
        x: (B, H, W, C)
    """
    x = tf.reshape(
        windows,
        shape=(-1, h // window_size, w // window_size, window_size, window_size, c),
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, shape=(-1, h, w, c))
    return x


def window_partition(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): Window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    b, h, w, c = tf.unstack(tf.shape(x))
    x = tf.reshape(
        x, shape=(-1, h // window_size, window_size, w // window_size, window_size, c)
    )
    x = tf.transpose(x, perm=(0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, shape=(-1, window_size, window_size, c))
    return windows