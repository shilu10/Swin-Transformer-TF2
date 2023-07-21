from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from ml_collections import ConfigDict


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