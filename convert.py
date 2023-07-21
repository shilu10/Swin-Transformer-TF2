from .utils import modify_tf_block
from .swin_transformer.model import SwinTransformer
from .swin_transformer.layers import *
from .swin_transformer.blocks import *
import numpy as np 
import os, sys, shutil
import tqdm 
import glob 
import pandas as pd 
import tensorflow as tf 
import tensorflow.keras as keras 
import argparse
import timm, transformers 
from typing import Dict, List
from transformers import SwinModel, SwinForImageClassification
import yaml 
from imutils import paths
from .base_config import get_base_config

def port_weights(model_type="swin_tiny_patch4_window7_224", include_top=True):
    print("Intializing the Tensorflow Model")
    
    # read the data from yaml file
    config_file_path = f"configs/{model_type}.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    config = get_base_config(
              input_size=(data.get("image_size"), data.get("image_size")),
              patch_size=data.get("patch_size"),
              embed_dim=data.get("embed_dim"),
              window_size=data.get("window_size"),
              nb_blocks=data.get("nb_blocks"),
              nb_heads=data.get("nb_heads"),
              include_top=include_top
        )
    
    tf_model = SwinTransformer(config)

    img_dim = data.get('image_size')
    dummy_input = np.zeros((1, img_dim, img_dim, 3))
    _ = tf_model(dummy_input)

    print('Loading the Pytorch model!!!')
    pt_model = SwinForImageClassification.from_pretrained(f"microsoft/swin-base-patch4-window7-224")
    pt_model.eval()

    # pt_model_dict
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: np.array(pt_model_dict[k]) for k in pt_model_dict.keys()}

    # main norm
    tf_model.layers[-3] = modify_tf_block(
          tf_model.layers[-3],
          pt_model_dict["swin.layernorm.weight"],
          pt_model_dict["swin.layernorm.bias"],

        )

    # patch embed layer's projection
    tf_model.layers[0].projection = modify_tf_block(
        tf_model.layers[0].projection,
        (pt_model_dict["swin.embeddings.patch_embeddings.projection.weight"]),
        (pt_model_dict["swin.embeddings.patch_embeddings.projection.bias"])
      )

    # patch embed layer's normalization
    tf_model.layers[0].norm = modify_tf_block(
        tf_model.layers[0].norm,
        (pt_model_dict["swin.embeddings.norm.weight"]),
        (pt_model_dict["swin.embeddings.norm.bias"])
      )

    if include_top:
      # classification layer
      tf_model.layers[-1] = modify_tf_block(
        tf_model.layers[-1],
        pt_model_dict["classifier.weight"],
        pt_model_dict["classifier.bias"],
      )

    # for swin layers
    for indx, stage in enumerate(tf_model.layers[2: len(config.nb_blocks)+2]):
      modify_swin_layer(stage, indx, pt_model_dict)

    model_name = model_type if include_top else model_type + "_fe"
    tf_model.save_weights(model_name + ".h5")
    print("Tensorflow model weights saved successfully at: ", model_name)


def modify_swin_layer(swin_layer, swin_layer_indx, pt_model_dict):

  for block_indx, block in enumerate(swin_layer.layers):
    # layer and block combined name
    pt_block_name = f"swin.encoder.layers.{swin_layer_indx}.blocks.{block_indx}"

    if isinstance(block, PatchMerging):

      norm_weight = (pt_model_dict[f"swin.encoder.layers.{swin_layer_indx}.downsample.norm.weight"]).transpose()
      norm_bias = (pt_model_dict[f"swin.encoder.layers.{swin_layer_indx}.downsample.norm.bias"]).transpose()

      block.norm.gamma.assign(tf.Variable(norm_weight))
      block.norm.beta.assign(tf.Variable(norm_bias))

      # reduction
      block.reduction = modify_tf_block(
          block.reduction,
          (pt_model_dict[f"swin.encoder.layers.{swin_layer_indx}.downsample.reduction.weight"]),
        )

    if isinstance(block, SwinTransformerBlock):
      # norm1
      norm_layer_name = pt_block_name + ".layernorm_before"
      block.norm1 = modify_tf_block(
          block.norm1,
          (pt_model_dict[ norm_layer_name + ".weight"]),
          (pt_model_dict[ norm_layer_name + ".bias"])
        )

      # norm2 
      norm_layer_name = pt_block_name + ".layernorm_after"
      block.norm2 = modify_tf_block(
          block.norm2,
          (pt_model_dict[ norm_layer_name + ".weight"]),
          (pt_model_dict[ norm_layer_name + ".bias"])
        )

      # window attn
      block.attn.relative_position_bias_table = modify_tf_block(
          block.attn.relative_position_bias_table,
          (pt_model_dict[pt_block_name + ".attention.self.relative_position_bias_table"]),
        )

      # relative_position_index
      block.attn.relative_position_index = modify_tf_block(
          block.attn.relative_position_index,
          (pt_model_dict[pt_block_name + ".attention.self.relative_position_index"]),
        )

      # qkv matrix
      q_weight = (pt_model_dict[pt_block_name + ".attention.self.query.weight"])
      k_weight = (pt_model_dict[pt_block_name + ".attention.self.key.weight"])
      v_weight = (pt_model_dict[pt_block_name + ".attention.self.value.weight"])

      qkv_weight = np.concatenate([q_weight, k_weight, v_weight])

      q_bias = (pt_model_dict[pt_block_name + ".attention.self.query.bias"])
      k_bias = (pt_model_dict[pt_block_name + ".attention.self.key.bias"])
      v_bias = (pt_model_dict[pt_block_name + ".attention.self.value.bias"])

      qkv_bias = np.concatenate([q_bias, k_bias, v_bias])

      block.attn.qkv = modify_tf_block(
          block.attn.qkv,
          qkv_weight,
          qkv_bias
        )

      # qkv projection
      block.attn.proj = modify_tf_block(
          block.attn.proj,
          (pt_model_dict[pt_block_name + ".attention.output.dense.weight"]),
          (pt_model_dict[pt_block_name + ".attention.output.dense.bias"])
        )
      
      # mlp 
      block.mlp.fc1 = modify_tf_block(
            block.mlp.fc1,
            (pt_model_dict[pt_block_name + ".intermediate.dense.weight"]),
            (pt_model_dict[pt_block_name + ".intermediate.dense.bias"])
          )
      
      block.mlp.fc2 = modify_tf_block(
            block.mlp.fc2,
            (pt_model_dict[pt_block_name + ".output.dense.weight"]),
            (pt_model_dict[pt_block_name + ".output.dense.bias"])
          )
