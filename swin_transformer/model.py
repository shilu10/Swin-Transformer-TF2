import tensorflow as tf 
from tensorflow import keras 
from ml_collections import ConfigDict 
from collections import * 
import collections
from .blocks import SwinTransformerStage
from .layers import PatchEmbed, norm_layer_factory, act_layer_factory
from typing import *
import numpy as np


class SwinTransformer(tf.keras.Model):

    def __init__(self, config: ConfigDict, *args, **kwargs):
        super(SwinTransformer, self).__init__(*args, **kwargs)
        self.config = config
        self.norm_layer = norm_layer_factory(config.norm_layer)

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
           config,
           name="patch_embedding"
        )
        self.drop = tf.keras.layers.Dropout(config.drop_rate)

        # Stochastic depth
        dpr = np.linspace(0.0, config.drop_path_rate, sum(config.nb_blocks))

        # Build stages
        self.stages = []
        nb_stages = len(config.nb_blocks)
        block_idx_to = 0
        for idx in range(nb_stages):
            block_idx_from = block_idx_to
            block_idx_to = block_idx_to + config.nb_blocks[idx]

            self.stages.append(
                SwinTransformerStage(
                    config=config,
                    input_size=(
                        config.patch_resolution[0] // (2**idx),
                        config.patch_resolution[1] // (2**idx),
                    ),
                    embed_dim=int(config.embed_dim * 2**idx),
                    nb_blocks=config.nb_blocks[idx],
                    nb_heads=config.nb_heads[idx],
                    drop_path_rate=dpr[block_idx_from:block_idx_to],
                    downsample=idx < nb_stages - 1,  # Don't downsample the last stage
                    name=f"layers/{idx}",
                )
            )

        self.norm = self.norm_layer(name="norm")
        self.pool = tf.keras.layers.GlobalAveragePooling1D()
        self.head = (
            tf.keras.layers.Dense(units=config.nb_classes, name="head")
            if config.nb_classes > 0
            else tf.keras.layers.Activation("linear")  # Identity layer
        )

    @property
    def feature_names(self) -> List[str]:
        names = ["patch_embedding"]
        k = 0
        for j in range(len(self.config.nb_blocks)):
            for _ in range(self.config.nb_blocks[j]):
                names.append(f"block_{k}")
                k += 1
            names.append(f"stage_{j}")
        names += ["features_all", "features", "logits"]
        return names

    @property
    def keys_to_ignore_on_load_missing(self) -> List[str]:
        names = []
        for j in range(len(self.config.nb_blocks)):
            for k in range(self.config.nb_blocks[j]):
                names.append(f"layers/{j}/blocks/{k}/attn_mask")
                names.append(f"layers/{j}/blocks/{k}/attn/relative_position_index")
        return names

    def forward_features(self, x, training=False, return_features=False):
        features = {}
        x = self.patch_embed(x, training=training)
        x = self.drop(x, training=training)
        features["patch_embedding"] = x

        block_idx = 0
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, training=training, return_features=return_features)
            if return_features:
                x, stage_features = x
                for k in range(self.config.nb_blocks[stage_idx]):
                    features[f"block_{block_idx}"] = stage_features[f"block_{k}"]
                    block_idx += 1
                features[f"stage_{stage_idx}"] = stage_features["features"]

        x = self.norm(x, training=training)
        features["features_all"] = x
        x = self.pool(x)
        features["features"] = x
        return (x, features) if return_features else x

    def call(self, x, training=False, return_features=False):
        features = {}
        x = self.forward_features(x, training, return_features)
        if return_features:
            x, features = x
        x = self.head(x)
        features["logits"] = x
        return (x, features) if return_features else x