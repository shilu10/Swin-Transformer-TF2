from utils import modify_tf_block
from swin_transformer.model import SwinTransformer
from swin_transformer.model import *
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
from transformers import SwinModel
import yaml 
from imutils import paths

def port_weights(model_type="swin_tiny_patch4_window7_224", include_top=True):
    print("Intializing the Tensorflow Model")
    
    # read the data from yaml file
    config_file_path = f"configs/{model_type}.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)
    
    tf_model = SwinTransformer(
        img_size = data.get("img_size"),
        patch_size = data.get("patch_size"),
        embed_dim = data.get("embed_dim"),
        depths = data.get("depths"), 
        num_heads = data.get("num_heads"),
        window_size = data.get('window_size'),
        include_top = include_top
    )

    img_dim = data.get('img_size')
    dummy_input = np.zeros((1, img_dim, img_dim, 3))
    _ = tf_model(dummy_input)

    print('Loading the Pytorch model!!!')
    timm_pt_model = timm.create_model(
        model_name = model_type,
        pretrained = True
    )
    timm_pt_model.eval()
    huggingface_pt_model = pt_model = SwinModel.from_pretrained(f"microsoft/{model_type.replace('_', '-')}")

    # weight dict of huggingface model
    np_state_dict = huggingface_pt_model.state_dict()
    pt_model_dict = {k: np_state_dict[k].numpy() for k in np_state_dict}

    # weight dict of timm model.
    timm_np_state_dict = timm_pt_model.state_dict()
    timm_pt_model_dict = {k: timm_np_state_dict[k].numpy() for k in timm_np_state_dict}

    del huggingface_pt_model
    del timm_pt_model

    # main norm layer
    tf_model.layers[-2] = modify_tf_block(
            tf_model.layers[-2],
            pt_model_dict["layernorm.weight"],
            pt_model_dict["layernorm.bias"],

        )

    # patch embed layer's projection

    tf_model.layers[0].proj = modify_tf_block(
        tf_model.layers[0].proj,
        np.array(pt_model_dict["embeddings.patch_embeddings.projection.weight"]),
        np.array(pt_model_dict["embeddings.patch_embeddings.projection.bias"])
    )

    # patch embed layer's normalization
    tf_model.layers[0].norm = modify_tf_block(
        tf_model.layers[0].norm,
        np.array(pt_model_dict["embeddings.norm.weight"]),
        np.array(pt_model_dict["embeddings.norm.bias"])
    )


    if include_top:
        # classification layer

        tf_model.layers[-1] = modify_tf_block(
                tf_model.layers[-1],
                timm_pt_model_dict["head.fc.weight"],
                timm_pt_model_dict["head.fc.bias"],
            )

    # for swin layers
    for i in range(len(data.get("depths"))):
        swin_layer = tf_model.layers[2 + i]
        modify_swin_layer(swin_layer, i, pt_model_dict)

    model_name = model_type if include_top else model_type + "_fe"
    tf_model.save_weights(model_name + ".h5")
    print("Tensorflow model weights saved successfully at: ", model_name)


def modify_swin_layer(swin_layer, swin_layer_indx, pt_model_dict):

  for block_indx, block in enumerate(swin_layer.layers):

    # layer and block combined name
    pt_block_name = f"encoder.layers.{swin_layer_indx}.blocks.{block_indx}"

    if isinstance(block, PatchMerging):

      norm_weight = np.array(pt_model_dict[f"encoder.layers.{swin_layer_indx}.downsample.norm.weight"]).transpose()
      norm_bias = np.array(pt_model_dict[f"encoder.layers.{swin_layer_indx}.downsample.norm.bias"]).transpose()

      block.norm.gamma.assign(tf.Variable(norm_weight))
      block.norm.beta.assign(tf.Variable(norm_bias))

      # reduction
      block.reduction = modify_tf_block(
          block.reduction,
          np.array(pt_model_dict[f"encoder.layers.{swin_layer_indx}.downsample.reduction.weight"]),
        )

    if isinstance(block, SwinTransformerBlock):
      n_norm = 1

      for inner_transformer_block in block.layers:

        # Normalization layer (norm1 and norm2)
        if isinstance(inner_transformer_block, LayerNormalization):
          if n_norm == 1:
            norm_layer_name = pt_block_name + ".layernorm_before"
          else:
            norm_layer_name = pt_block_name + ".layernorm_after"

         # print(inner_transformer_block)

          inner_transformer_block = modify_tf_block(
              inner_transformer_block,
              np.array(pt_model_dict[ norm_layer_name + ".weight"]),
              np.array(pt_model_dict[ norm_layer_name + ".bias"])
          )
          n_norm += 1

        # window attention layer:
        if isinstance(inner_transformer_block, WindowAttention):
          # relative position bias table
          inner_transformer_block.relative_position_bias_table = modify_tf_block(
            inner_transformer_block.relative_position_bias_table,
            np.array(pt_model_dict[pt_block_name + ".attention.self.relative_position_bias_table"]),
          )

          # relative_position_index
          inner_transformer_block.relative_position_index = modify_tf_block(
            inner_transformer_block.relative_position_index,
            np.array(pt_model_dict[pt_block_name + ".attention.self.relative_position_index"]),
          )

          # qkv matrix
          q_weight = np.array(pt_model_dict[pt_block_name + ".attention.self.query.weight"])
          k_weight = np.array(pt_model_dict[pt_block_name + ".attention.self.key.weight"])
          v_weight = np.array(pt_model_dict[pt_block_name + ".attention.self.value.weight"])

          qkv_weight = np.concatenate([q_weight, k_weight, v_weight])

          q_bias = np.array(pt_model_dict[pt_block_name + ".attention.self.query.bias"])
          k_bias = np.array(pt_model_dict[pt_block_name + ".attention.self.key.bias"])
          v_bias = np.array(pt_model_dict[pt_block_name + ".attention.self.value.bias"])

          qkv_bias = np.concatenate([q_bias, k_bias, v_bias])

          inner_transformer_block.qkv = modify_tf_block(
            inner_transformer_block.qkv,
            qkv_weight,
            qkv_bias
          )

          # qkv projection
          inner_transformer_block.proj = modify_tf_block(
            inner_transformer_block.proj,
            np.array(pt_model_dict[pt_block_name + ".attention.output.dense.weight"]),
            np.array(pt_model_dict[pt_block_name + ".attention.output.dense.bias"])
          )

          # mlp layer
        if isinstance(inner_transformer_block, MLP):
          # fc1
          inner_transformer_block.fc1 = modify_tf_block(
            inner_transformer_block.fc1,
            np.array(pt_model_dict[pt_block_name + ".intermediate.dense.weight"]),
            np.array(pt_model_dict[pt_block_name + ".intermediate.dense.bias"])
          )

          # fc2
          inner_transformer_block.fc2 = modify_tf_block(
            inner_transformer_block.fc2,
            np.array(pt_model_dict[pt_block_name + ".output.dense.weight"]),
            np.array(pt_model_dict[pt_block_name + ".output.dense.bias"])
          )

