from .utils import modify_tf_block
from swin_transformer.models import SwinTransformer
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


parser = argparse.ArgumentParser(description="Conversion of the PyTorch pre-trained Swin weights to TensorFlow.")
parser.add_argument("-m", 
                    "--model-type", 
                    default="swin_tiny_patch4_window7_224", 
                    type=str, 
                    help="Name of the Swin model variant."
                )

parser.add_argument(
        "-it",
        "--include_top",
        action="store_true",
        help="If we don't need the classification outputs.",
    )

args = parser.parse_args()


def main(args):
    print("Intializing the Tensorflow Model")
     


if __name__ == '__main__':
    main(args)