from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from .utils import get_initializer
from .factory import act_layer_factory, norm_layer_factory
from collections import * 
import collections
from ml_collections import ConfigDict
from typing import *
import numpy as np


class PatchEmbed(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config: ConfigDict, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        image_size = config.image_size
        patch_size = config.patch_size
        projection_dim = config.projection_dim
        n_channels = config.n_channels

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = ((image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]))
        act_layer = norm_layer_factory(config.norm_layer)

        patches_resolution = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution

        # calculation of num of patches
        self.num_patches = num_patches
        self.config = config
        self.image_size = image_size
        self.n_channels = n_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size

        # patch generator
        self.projection = tf.keras.layers.Conv2D(
            kernel_size=patch_size,
            strides=patch_size,
            data_format="channels_last",
            filters=projection_dim,
            padding="valid",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection"
        )

        self.norm = act_layer()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shape = tf.shape(x)
        batch_size, height, width, n_channel = shape[0], shape[1], shape[2], shape[3]

        projection = self.projection(x)
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, self.num_patches, -1))

        embeddings = self.norm(embeddings)

        return embeddings

    def get_config(self):
        config = super(PatchEmbed, self).get_config()
        config["image_size"] = self.image_size
        config["projection_dim"] = self.projection_dim
        config["patch_size"] = self.patch_size
        config["n_channels"] = self.n_channels
        return config

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.config.embed_dim * self.n_channels * (self.patch_size[0] * self.patch_size[1])
        
        return flops
