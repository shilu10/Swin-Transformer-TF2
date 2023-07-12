from .utils import modify_tf_block
from .swin_transformer.model import SwinTransformer
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

    print('Loading the Pytorch model!!!')
    timm_pt_model = timm.create_model(
        model_name = model_type,
        pretrained = True
    )
    huggingface_pt_model = pt_model = SwinModel.from_pretrained(f"microsoft/{model_type.replace('_', "-")}")

    # weight dict of huggingface model
    np_state_dict = huggingface_pt_model.state_dict()
    pt_model_dict = {k: np_state_dict[k].numpy() for k in np_state_dict}

    # weight dict of timm model.
    timm_np_state_dict = timm_pt_model.state_dict()
    timm_pt_model_dict = {k: timm_np_state_dict[k].numpy() for k in timm_np_state_dict}

    del huggingface_pt_model
    del timm_pt_model

    print(timm_pt_model_dict.keys())
    print(pt_model_dict.keys())