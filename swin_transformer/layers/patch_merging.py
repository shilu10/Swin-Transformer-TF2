from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from ml_collections import ConfigDict


class PatchMerging(tf.keras.layers.Layer):
    """ Patch Merging Layer.
        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(
        self,
        config: ConfigDict,
        input_size: Tuple[int, int],
        embed_dim: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config
        self.input_size = input_size
        self.norm_layer = norm_layer_factory(config.norm_layer)

        self.reduction = tf.keras.layers.Dense(
            units=2 * embed_dim, use_bias=False, name="reduction"
        )
        self.norm = self.norm_layer(name="norm")

    def call(self, x, training=False):
        h, w = self.input_size
        b, l, c = tf.unstack(tf.shape(x))

        x = tf.reshape(x, shape=(-1, h, w, c))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat((x0, x1, x2, x3), axis=-1)
        x = tf.reshape(x, shape=(-1, (h // 2) * (w // 2), 4 * c))

        x = self.norm(x, training=training)
        x = self.reduction(x)
        return x



    