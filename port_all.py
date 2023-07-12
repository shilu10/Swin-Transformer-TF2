import os 
import shutil 
import glob 
import numpy as np 
import yaml 
from imutils import paths 
from porting import port_weights

# all config files 

def main():

    try:
        config_file_paths = list(paths.list_files("configs/"))
        for config_file_path in config_file_paths:
            # porting all model types from pytorch to tensorflow

            model_type = config_file_path.split("/")[-1].split(".")[0]
            print(f"Processing the  model type: {model_type}")

            port_weights(
                model_type=model_type
                include_top=True
            )    

    except Exception as err:
        return err



