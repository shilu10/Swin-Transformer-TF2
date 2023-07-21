from ml_collections import ConfigDict 
from typing import * 

def get_base_config(input_size: Tuple = (224, 224), 
                    patch_size: int = 4,
                    embed_dim: int = 96,
                    window_size: int = 7,
                    nb_blocks: List = [2, 2, 6, 2],
                    nb_heads: List = [3, 6, 12, 24],
                    drop_rate: float = 0.0, 
                    drop_path_rate: float = 0.0, 
                    nb_classes: int = 1000,
                    include_top: bool = True,
                    attn_drop_rate: float = 0.0,
                ):

    config = ConfigDict() 
    config.patch_size = patch_size
    config.projection_dim = projection_dim
    config.window_size = window_size 
    config.nb_blocks = nb_blocks
    config.nb_heads = nb_heads
    config.nb_classes = nb_classes
    config.drop_rate = drop_rate 
    config.drop_path_rate = drop_path_rate
    config.attn_drop_rate = attn_drop_rate
    config.include_top = include_top
    config.input_size = input_size
    config.image_size = config.input_size

    
    # common configs
    config.initializer_range = 0.1
    config.patch_resolution = (config.image_size[0] // config.patch_size, config.image_size[1] // config.patch_size)
    config.act_layer = 'gelu'
    config.norm_layer = 'layer_norm'

    return config.lock()