from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.layers import * 
from tensorflow.keras import Model 
from .swin_transformer_block import SwinTransformerBlock
from ml_collections import ConfigDict
from ..layers import PatchMerging
from typing import *
import numpy as np


class SwinTransformerStage(tf.keras.Model):
    def __init__(
        self,
        config: ConfigDict,
        input_size: Tuple[int, int],
        embed_dim: int,
        nb_blocks: int,
        nb_heads: int,
        drop_path_rate: np.ndarray,
        downsample: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.config = config

        self.blocks = [
            SwinTransformerBlock(
                config=config,
                input_size=input_size,
                embed_dim=embed_dim,
                nb_heads=nb_heads,
                drop_path_rate=drop_path_rate[idx],
                shift_size=0 if idx % 2 == 0 else config.window_size // 2,
                name=f"blocks/{idx}",
            )
            for idx in range(nb_blocks)
        ]
        if downsample:
            self.downsample = PatchMerging(
                config=config, input_size=input_size, embed_dim=embed_dim, name="downsample"
            )
        else:
            self.downsample = tf.keras.layers.Activation("linear")

    def call(self, x, training=False, return_features=False):
        features = {}
        for j, block in enumerate(self.blocks):
            x = block(x, training=training)
            features[f"block_{j}"] = x

        x = self.downsample(x, training=training)
        features["features"] = x
        return (x, features) if return_features else x