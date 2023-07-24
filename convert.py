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
import torch

def port_weights(model_type="swin_tiny_patch4_window7_224", 
                model_savepath =".", 
                include_top=True
              ):

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
    #pt_model = SwinForImageClassification.from_pretrained(f"microsoft/{model_type.replace('_', '-')}")
    #pt_model.eval()

    # pt_model_dict
    url = model_url[model_type]
    pt_model_dict = torch.hub.load_state_dict_from_url(url, map_location = "cpu", progress = True, check_hash = True)
    pt_model_dict = {k: np.array(pt_model_dict[k]) for k in pt_model_dict.keys()}

    # main norm
    tf_model.layers[-3] = modify_tf_block(
      tf_model.layers[-3],
      pt_model_dict["norm.weight"],
      pt_model_dict["norm.bias"],

    )

    # patch embed layer's projection
    tf_model.layers[0].projection = modify_tf_block(
      tf_model.layers[0].projection,
      (pt_model_dict["patch_embed.proj.weight"]),
      (pt_model_dict["patch_embed.proj.bias"])
    )

    # patch embed layer's normalization
    tf_model.layers[0].norm = modify_tf_block(
      tf_model.layers[0].norm,
      (pt_model_dict["patch_embed.norm.weight"]),
      (pt_model_dict["patch_embed.norm.bias"])
    )

    if include_top:
      # classification layer
      tf_model.layers[-1] = modify_tf_block(
        tf_model.layers[-1],
        pt_model_dict["head.weight"],
        pt_model_dict["head.bias"],
      )

    # for swin layers
    for indx, stage in enumerate(tf_model.layers[2: len(config.nb_blocks)+2]):
      modify_swin_layer(stage, indx, pt_model_dict)

    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized at: {save_path}...")


def modify_swin_layer(swin_layer, swin_layer_indx, pt_model_dict):

  for block_indx, block in enumerate(swin_layer.layers):
    print(isinstance(block,SwinTransformerBlock), block)

    # layer and block combined name
    pt_block_name = f"layers.{swin_layer_indx}.blocks.{block_indx}"

    if isinstance(block, PatchMerging):

      norm_weight = (pt_model_dict[f"layers.{swin_layer_indx}.downsample.norm.weight"]).transpose()
      norm_bias = (pt_model_dict[f"layers.{swin_layer_indx}.downsample.norm.bias"]).transpose()

      block.norm.gamma.assign(tf.Variable(norm_weight))
      block.norm.beta.assign(tf.Variable(norm_bias))

      # reduction
      block.reduction = modify_tf_block(
          block.reduction,
          (pt_model_dict[f"layers.{swin_layer_indx}.downsample.reduction.weight"]),
        )

    if isinstance(block, SwinTransformerBlock):
      # norm1
      norm_layer_name = pt_block_name + ".norm1"
      block.norm1 = modify_tf_block(
          block.norm1,
          (pt_model_dict[ norm_layer_name + ".weight"]),
          (pt_model_dict[ norm_layer_name + ".bias"])
        )

      # norm2
      norm_layer_name = pt_block_name + ".norm2"
      block.norm2 = modify_tf_block(
          block.norm2,
          (pt_model_dict[ norm_layer_name + ".weight"]),
          (pt_model_dict[ norm_layer_name + ".bias"])
        )

      # window attn

      block.attn.relative_position_bias_table = modify_tf_block(
          block.attn.relative_position_bias_table,
          (pt_model_dict[pt_block_name + ".attn.relative_position_bias_table"]),
        )

      # relative_position_index
      block.attn.relative_position_index = modify_tf_block(
          block.attn.relative_position_index,
          (pt_model_dict[pt_block_name + ".attn.relative_position_index"]),
        )

      # qkv matrix


      block.attn.qkv = modify_tf_block(
          block.attn.qkv,
          pt_model_dict[pt_block_name + ".attn.qkv.weight"],
          pt_model_dict[pt_block_name + ".attn.qkv.bias"]
        )

      # qkv projection
      block.attn.proj = modify_tf_block(
          block.attn.proj,
          (pt_model_dict[pt_block_name + ".attn.proj.weight"]),
          (pt_model_dict[pt_block_name + ".attn.proj.bias"])
        )

      # mlp
      block.mlp.fc1 = modify_tf_block(
            block.mlp.fc1,
            (pt_model_dict[pt_block_name + ".mlp.fc1.weight"]),
            (pt_model_dict[pt_block_name + ".mlp.fc1.bias"])
          )

      block.mlp.fc2 = modify_tf_block(
            block.mlp.fc2,
            (pt_model_dict[pt_block_name + ".mlp.fc2.weight"]),
            (pt_model_dict[pt_block_name + ".mlp.fc2.bias"])
          )

# url paths
model_url = {
  "swin_tiny_patch4_window7_224": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
  "swin_small_patch4_window7_224": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
  "swin_base_patch4_window7_224": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
  "swin_base_patch4_window12_384": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
  "swin_small_patch4_window7_224_22kto1k": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22kto1k_finetune.pth',
  "swin_base_patch4_window7_224_22kto1k": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22kto1k.pth',
  "swin_base_patch4_window12_384_22kto1k": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22kto1k.pth',
  "swin_large_patch4_window7_224_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22kto1k.pth",
  "swin_large_patch4_window12_384_22kto1k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth"
  
}
