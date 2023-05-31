class PatchMerging(keras.layers.Layer): 
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization):
        super(PatchMerging, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    
    def call(self, x): 
        """
        x: B, H * W, C
        """
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        x = tf.reshape(x, (B, H, W, C))
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        x = tf.concat([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = tf.reshape(x, (B, -1, 4 * C))# B H/2*W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x
