from ml_collections import ConfigDict 
from typing import * 

def get_base_config(image_size: int = 224, 
                    patch_size: int = 4,
                    projection_dim: int = 96,
                    window_size: int = 7,
                    depths: List = [2, 2, 6, 2],
                    num_heads: List = [3, 6, 12, 24],
                    drop_rate: float = 0.0, 
                    drop_path_rate: float = 0.0, 
                    
                    
            ):
    pass 