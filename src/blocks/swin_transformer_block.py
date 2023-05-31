class SwinTransformerBlock(keras.Model):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer='gelu',
                 norm_layer=LayerNormalization): 
    
        super(SwinTransformerBlock, self).__init__()
        self.dim = dim 
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio 
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        self.norm1 = norm_layer(epsilon=1e-5, name=f'norm1')

        self.attn = WindowAttention(
                        dim,
                        window_size=self.window_size,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop,
                        proj_drop=drop
                    )

        self.drop_path = DropPath(
            drop_path if (drop_path) > 0. else 0.
        )
        self.norm2 = norm_layer(epsilon=1e-5, name=f'norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MLP(input_dims=None,
                        hidden_neurons=mlp_hidden_dim,
                        output_neurons=None,
                        act_type='gelu',
                        dropout_rate=drop,
                        prefix='swin_transformer'
                      )
        if self.shift_size > 0:
                H, W = self.input_resolution
                img_mask = np.zeros([1, H, W, 1])
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1

                img_mask = tf.convert_to_tensor(img_mask)
                mask_windows = window_partition(img_mask, self.window_size)
                mask_windows = tf.reshape(
                    mask_windows, shape=[-1, self.window_size * self.window_size])
                attn_mask = tf.expand_dims(
                    mask_windows, axis=1) - tf.expand_dims(mask_windows, axis=2)
                attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
                attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
                self.attn_mask = tf.Variable(
                    initial_value=attn_mask, trainable=False)
        else:
            self.attn_mask = None

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (B, H, W, C))
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = utils.window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        if not return_attns:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        else:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=self.attn_mask, return_attns=True
            )  # [num_win*B, window_size*window_size, C]
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = utils.window_reverse(
            attn_windows, self.window_size, H, W
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x
        x = tf.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if return_attns:
            return x, attn_scores
        else:
            return x
        
