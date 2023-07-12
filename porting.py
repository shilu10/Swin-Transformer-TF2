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


def main(model_type="swin_tiny_patch4_window7_224", include_top=True):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    swin_trans = SwinTransformer()
    print("Intializing the Tensorflow Model", swin_trans)


