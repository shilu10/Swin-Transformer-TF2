from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.layers import * 
from tensorflow.keras import Model 
from ..layers import DropPath, WindowAttention
from .utils import window_partition, window_reverse
from ..layers import MLP
from ml_collections import ConfigDict
from ..layers import norm_layer_factory
from typing import *
import numpy as np


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        config: ConfigDict,
        input_size: Tuple[int, int],
        embed_dim: int,
        nb_heads: int,
        drop_path_rate: float,
        shift_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.input_size = input_size
        self.shift_size = shift_size
        self.norm_layer = norm_layer_factory(config.norm_layer)
        self.window_size = config.window_size

        # If the image resolution is smaller than the window size, there is no point
        # shifting windows, since we already capture the global context in that case.
        if min(self.input_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_size)

        self.norm1 = self.norm_layer(name="norm1")
        self.attn = WindowAttention(
            config=config,
            embed_dim=embed_dim,
            nb_heads=nb_heads,
            name="attn",
        )
        self.attn_mask = None
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.norm2 = self.norm_layer(name="norm2")
        self.mlp = MLP(
            hidden_dim=int(embed_dim * config.mlp_ratio),
            projection_dim=embed_dim,
            drop_rate=config.drop_rate,
            act_layer=config.act_layer,
            name="mlp",
        )

    def build(self, input_shape):
        h, w = self.input_size
        window_size = self.window_size
        shift_size = self.shift_size

        if shift_size > 0:
            img_mask = np.zeros([1, h, w, 1])
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_windows = window_partition(img_mask, window_size)
            mask_windows = tf.reshape(mask_windows, shape=(-1, window_size**2))
            attn_mask = tf.expand_dims(mask_windows, axis=1) - tf.expand_dims(
                mask_windows, axis=2
            )
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            # Only the non-trivial attention mask is a Variable, because in the PyTorch
            # model, only the non-trivial attention mask exists. The trivial maks
            # is replaced by None and if-statements in the call function.
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False, name="attn_mask"
            )
        else:
            # Attention mask is applied additively, so zero-mask has no effect
            # Broadcasting will take care of mapping it to the correct dimensions.
            self.attn_mask = tf.Variable(
                initial_value=tf.zeros((1,)), trainable=False, name="attn_mask"
            )

    def call(self, x, training=False):
        window_size = self.window_size
        shift_size = self.shift_size

        h, w = self.input_size
        b, l, c = tf.unstack(tf.shape(x))

        shortcut = x
        x = self.norm1(x, training=training)
        x = tf.reshape(x, shape=(-1, h, w, c))

        # Cyclic shift (Identify, if shift_size == 0)
        shifted_x = tf.roll(x, shift=(-shift_size, -shift_size), axis=[1, 2])

        # Partition windows
        x_windows = window_partition(shifted_x, window_size)
        x_windows = tf.reshape(x_windows, shape=(-1, window_size**2, c))

        # W-MSA/SW-MSA
        attn_windows = self.attn([x_windows, self.attn_mask])

        # Merge windows
        attn_windows = tf.reshape(attn_windows, shape=(-1, window_size, window_size, c))
        shifted_x = window_reverse(attn_windows, window_size, h, w, c)

        # Reverse cyclic shift
        x = tf.roll(shifted_x, shift=(shift_size, shift_size), axis=(1, 2))
        x = tf.reshape(x, shape=[-1, h * w, c])

        # Residual connection
        x = self.drop_path(x, training=training)
        x = x + shortcut

        # MLP
        shortcut = x
        x = self.norm2(x, training=training)
        x = self.mlp(x, training=training)
        x = self.drop_path(x, training=training)
        x = x + shortcut

        return x

    def get_config(self):
        config = super(SwinTransformerBlock, self).get_config()
        config["input_size"] = self.input_size
        config["shift_size"] = self.shift_size
        config["window_size"] = self.window_size
        return config