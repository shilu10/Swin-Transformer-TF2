from tensorflow import keras 
import tensorflow as tf 
import numpy as np
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from ml_collections import ConfigDict
from typing import *


class WindowAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        config: ConfigDict,
        embed_dim: int,
        nb_heads: int,
        **kwargs,
    ):
        super(WindowAttention, self).__init__(**kwargs)
        self.config = config
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads

        self.qkv = tf.keras.layers.Dense(
            embed_dim * 3, use_bias=config.qkv_bias, name="qkv"
        )
        self.attn_drop = tf.keras.layers.Dropout(config.attn_drop_rate)
        self.proj = tf.keras.layers.Dense(embed_dim, name="proj")
        self.proj_drop = tf.keras.layers.Dropout(config.drop_rate)

    def build(self, input_shape):
        window_size = self.config.window_size

        # The weights have to be created inside the build() function for the right
        # name scope to be set.
        self.relative_position_bias_table = self.add_weight(
            name="relative_position_bias_table",
            shape=((2 * window_size - 1) * (2 * window_size - 1), self.nb_heads),
            initializer=tf.initializers.Zeros(),
            trainable=True,
        )

        coords_h = np.arange(window_size)
        coords_w = np.arange(window_size)
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose((1, 2, 0))
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1).astype(np.int64)
        self.relative_position_index = tf.Variable(
            name="relative_position_index",
            initial_value=tf.convert_to_tensor(relative_position_index),
            trainable=False,
        )

    def call(self, inputs, training=False):
        nb_heads = self.nb_heads
        window_size = self.config.window_size

        # Inputs are the batch and the attention mask
        x, mask = inputs[0], inputs[1]
        _, n, c = tf.unstack(tf.shape(x))

        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, shape=(-1, n, 3, nb_heads, c // nb_heads))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)

        scale = (self.embed_dim // nb_heads) ** -0.5
        q = q * scale
        attn = q @ tf.transpose(k, perm=(0, 1, 3, 2))
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, shape=(-1,)),
        )
        relative_position_bias = tf.reshape(
            relative_position_bias,
            shape=(window_size**2, window_size**2, -1),
        )
        relative_position_bias = tf.transpose(relative_position_bias, perm=(2, 0, 1))
        attn = attn + tf.expand_dims(relative_position_bias, axis=0)

        nw = mask.get_shape()[0]  # tf.shape(mask)[0]
        mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
        mask = tf.cast(mask, attn.dtype)
        attn = tf.reshape(attn, shape=(-1, nw, nb_heads, n, n)) + mask
        attn = tf.reshape(attn, shape=(-1, nb_heads, n, n))
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)

        x = tf.transpose((attn @ v), perm=(0, 2, 1, 3))
        x = tf.reshape(x, shape=(-1, n, c))
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

    def get_config(self):
        config = super(WindowAttention, self).get_config()
        config["nb_heads"] = self.nb_heads
        config["embed_dim"] = self.embed_dim

        return config

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.embed_dim * 3 * self.embed_dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.nb_heads * N * (self.embed_dim // self.nb_heads) * N
        #  x = (attn @ v)
        flops += self.nb_heads * N * N * (self.embed_dim // self.nb_heads)
        # x = self.proj(x)
        flops += N * self.embed_dim * self.embed_dim
        return flops



class WindowAttentionV2(tf.keras.layers.Layer):
    def __init__(
        self,
        config: ConfigDict,
        embed_dim: int,
        nb_heads: int,
        pretrained_window_size: tuple,
        **kwargs,
    ):
        super(WindowAttentionV2, self).__init__(**kwargs)

        window_size = config.window_size
        self.window_size = window_size if not isinstance(window_size, int) else [window_size, window_size]

        self.pretrained_window_size = (pretrained_window_size if not isinstance(pretrained_window_size, int)
                                            else [pretrained_window_size, pretrained_window_size])


        self.config = config
        self.embed_dim = embed_dim
        self.nb_heads = nb_heads

        self.attn_drop = tf.keras.layers.Dropout(config.attn_drop_rate)
        self.proj_drop = tf.keras.layers.Dropout(config.drop_rate)

    def build(self, input_shape):
        self.logit_scale = self.add_weight("logit_scale",
                                           shape = [self.nb_heads, 1, 1],
                                           initializer = tf.keras.initializers.Constant(np.log(10.)),
                                           trainable = True)

        self.cpb_mlp1 = tf.keras.layers.Dense(512,
                                              use_bias = True,
                                              activation = tf.keras.activations.relu,
                                              name = "cpb_mlp_0")

        self.cpb_mlp2 = tf.keras.layers.Dense(self.nb_heads,
                                              use_bias = False,
                                              name = "cpb_mlp_2")

        relative_coords_h = np.arange(-(self.window_size[0] - 1), self.window_size[0])
        relative_coords_w = np.arange(-(self.window_size[1] - 1), self.window_size[1])
        relative_coords_table = np.expand_dims(np.transpose(np.stack(np.meshgrid(relative_coords_h, relative_coords_w, indexing = "ij")), [1, 2, 0]), axis = 0) #1, 2*Wh-1, 2*Ww-1, 2
        if 0 < self.pretrained_window_size[0]:
            relative_coords_table[:, :, :, 0] = relative_coords_table[:, :, :, 0] / (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] = relative_coords_table[:, :, :, 1] / (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] = relative_coords_table[:, :, :, 0] / (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] = relative_coords_table[:, :, :, 1] / (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = np.sign(relative_coords_table) * np.log2(np.abs(relative_coords_table) + 1.0) / np.log2(8)
        self.relative_coords_table = self.add_weight("relative_coords_table",
                                                     shape = np.shape(relative_coords_table),
                                                     initializer = tf.keras.initializers.Constant(relative_coords_table),
                                                     trainable = True)

        coords_h = np.arange(self.window_size[0])
        coords_w = np.arange(self.window_size[1])
        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing = "ij")) #2, Wh, Ww
        coords = np.reshape(coords, [2, -1])
        relative_coords = np.expand_dims(coords, axis = -1) - np.expand_dims(coords, axis = -2) #2, Wh * Ww, Wh * Ww
        relative_coords = np.transpose(relative_coords, [1, 2, 0]) #Wh * Ww, Wh * Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1 #shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = np.sum(relative_coords, -1)
        self.relative_position_index = tf.Variable(tf.convert_to_tensor(relative_position_index),
                                                  trainable = False,
                                                  name= "relative_position_index")

        self.qkv = tf.keras.layers.Dense(self.embed_dim * 3, use_bias = False, name = "qkv")
        self.proj = tf.keras.layers.Dense(self.embed_dim, use_bias = True, name = "proj")

        if self.config.qkv_bias:
            self.q_bias = self.add_weight("q_bias",
                                          shape = [self.embed_dim],
                                          initializer = tf.keras.initializers.Zeros,
                                          trainable = True)

            self.v_bias = self.add_weight("v_bias",
                                          shape = [self.embed_dim],
                                          initializer = tf.keras.initializers.Zeros,
                                          trainable = True)

    def call(self, inputs, training=False):
        nb_heads = self.nb_heads
        window_size = self.config.window_size

        # Inputs are the batch and the attention mask
        x, mask = inputs[0], inputs[1]
        _, n, c = tf.unstack(tf.shape(x))

        qkv = self.qkv(x)
        if self.config.qkv_bias:
            qkv_bias = tf.concat([self.q_bias, tf.stop_gradient(tf.zeros_like(self.v_bias)), self.v_bias], axis = 0)
            qkv = tf.nn.bias_add(qkv, qkv_bias)

        qkv = tf.reshape(qkv, shape=(-1, n, 3, nb_heads, c // nb_heads))
        qkv = tf.transpose(qkv, perm=(2, 0, 3, 1, 4))
        q, k, v = tf.unstack(qkv)

        # cosine attention
        attn = tf.linalg.normalize(q, axis = -1)[0] @ tf.transpose(tf.linalg.normalize(k, axis = -1)[0], [0, 1, 3, 2])
        #attn = tf.linalg.normalize(q, axis = -1)[0] @ tf.transpose(tf.linalg.normalize(k, axis = -1)[0], [0, 1, 3, 2])
        #attn = tf.math.l2_normalize(q, axis = -1)[0] @ tf.transpose(tf.math.l2_normalize(k, axis = -1)[0], perm=(0, 1, 3, 2))

        logit_scale = tf.exp(tf.minimum(self.logit_scale, np.log(1. / 0.01)))
        attn = attn * logit_scale


        relative_position_bias_table = tf.reshape(self.cpb_mlp2(self.cpb_mlp1(self.relative_coords_table)), [-1, self.nb_heads])
        relative_position_bias = tf.gather(relative_position_bias_table, tf.reshape(self.relative_position_index, [-1]))
        relative_position_bias = tf.reshape(relative_position_bias, [self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1]) # Wh*Ww,Wh*Ww,nH
        relative_position_bias = tf.transpose(relative_position_bias, [2, 0, 1]) # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * tf.nn.sigmoid(relative_position_bias)
        attn = attn + tf.expand_dims(relative_position_bias, axis = 0)

        if mask is not None:
            n_window =  mask.get_shape()[0]
            attn = tf.reshape(attn, [-1, n_window, self.nb_heads, n, n]) + tf.cast(tf.expand_dims(tf.expand_dims(mask, axis = 1), axis = 0), attn.dtype)
            attn = tf.reshape(attn, [-1, self.nb_heads, n, n])
        attn = tf.nn.softmax(attn, axis = -1)

        attn = self.attn_drop(attn)

        out = tf.reshape(tf.transpose((attn @ v), [0, 2, 1, 3]), shape=[-1, n, c])
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def get_config(self):
        config = super(WindowAttention, self).get_config()
        config["nb_heads"] = self.nb_heads
        config["embed_dim"] = self.embed_dim

        return config

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.embed_dim * 3 * self.embed_dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.nb_heads * N * (self.embed_dim // self.nb_heads) * N
        #  x = (attn @ v)
        flops += self.nb_heads * N * N * (self.embed_dim // self.nb_heads)
        # x = self.proj(x)
        flops += N * self.embed_dim * self.embed_dim
        return flops