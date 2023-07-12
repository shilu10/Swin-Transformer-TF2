from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
import os 
from tensorflow.keras.layers import Dense, Input, UpSampling2D, Conv2DTranspose, Conv2D, add, Add,\
                    Lambda, Concatenate, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, \
                    Add, LayerNormalization, Activation, LeakyReLU, SeparableConv2D
from tensorflow.keras import Model
import collections.abc
from typing import Tuple, Union
from .utils import to_ntuple
from tensorflow.keras.layers import *
from tensorflow.keras import layers


def window_partition(x: tf.Tensor, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    windows = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(windows, (-1, window_size, window_size, C))
    return windows


def window_reverse(windows: tf.Tensor, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = tf.shape(windows)[0] // tf.cast(
        H * W / window_size / window_size, dtype="int32"
    )

    x = tf.reshape(
        windows,
        (
            B,
            H // window_size,
            W // window_size,
            window_size,
            window_size,
            -1,
        ),
    )
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    xx, yy = tf.meshgrid(range(win_h), range(win_w))
    coords = tf.stack([yy, xx], axis=0)  # [2, Wh, Ww]
    coords_flatten = tf.reshape(coords, [2, -1])  # [2, Wh*Ww]

    relative_coords = (
        coords_flatten[:, :, None] - coords_flatten[:, None, :]
    )  # [2, Wh*Ww, Wh*Ww]
    relative_coords = tf.transpose(
        relative_coords, perm=[1, 2, 0]
    )  # [Wh*Ww, Wh*Ww, 2]

    xx = (relative_coords[:, :, 0] + win_h - 1) * (2 * win_w - 1)
    yy = relative_coords[:, :, 1] + win_w - 1
    relative_coords = tf.stack([xx, yy], axis=-1)

    return tf.reduce_sum(relative_coords, axis=-1)  # [Wh*Ww, Wh*Ww]


def drop_path(inputs, drop_prob, is_training):
    if (not is_training) or (drop_prob == 0.):
        return inputs

    # Compute keep_prob
    keep_prob = 1.0 - drop_prob

    # Compute drop_connect tensor
    random_tensor = keep_prob
    shape = (tf.shape(inputs)[0],) + (1,) * \
        (len(tf.shape(inputs)) - 1)
    random_tensor += tf.random.uniform(shape, dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


class MLP(tf.keras.layers.Layer):
    """
        this class is a implementation of the mlp block described in the swin transformer paper, which contains
        2 fully connected layer with GelU activation.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        """
            Params:
                input_neurons(dtype: int)   : input dimension for the mlp block, it needed only for .summary() method.
                hidden_neurons(dtype: int)  : number of neurons in the hidden
                                              layer(fully connected layer).
                output_neurons(dtype: iny)  ; number of neurons in the last
                                              layer(fully connected layer) of mlp.
                act_type(type: str)         ; type of activation needed. in paper, GeLU is used.
                dropout_rate(dtype: float)  : dropout rate in the dropout layer.
                prefix(type: str)           : used for the naming the layers.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Dense(hidden_features, name=f'mlp/fc1')
        self.fc2 = Dense(out_features, name=f'mlp/fc2')
        self.drop = Dropout(drop)

    def call(self, x):
        x = self.fc1(x)
        x = tf.keras.activations.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(layers.Layer):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        head_dim (int): Number of channels per head (dim // num_heads if not set)
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, head_dim=None, window_size=7,
                     qkv_bias=True, attn_drop=0.0, proj_drop=0.0, **kwargs):

        super(WindowAttention, self).__init__(**kwargs)

        self.dim = dim
        self.window_size = (
            window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size)
        )  # Wh, Ww
        self.win_h, self.win_w = self.window_size
        self.window_area = self.win_h * self.win_w
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.attn_dim = self.head_dim * num_heads
        self.scale = self.head_dim ** -0.5

        # get pair-wise relative position index for each token inside the window
        self.relative_position_index = get_relative_position_index(
            self.win_h, self.win_w
        )

        self.qkv = layers.Dense(
            self.attn_dim * 3, use_bias=qkv_bias, name="attention_qkv"
        )
        self.attn_drop = layers.Dropout(attn_drop)
        self.proj = layers.Dense(dim, name="attention_projection")
        self.proj_drop = layers.Dropout(proj_drop)

    def build(self, input_shape):
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * self.win_h - 1) * (2 * self.win_w - 1), self.num_heads),
            initializer="zeros",
            trainable=True,
            name="relative_position_bias_table",
        )
        super().build(input_shape)

    def _get_rel_pos_bias(self) -> tf.Tensor:
        relative_position_bias = tf.gather(
            self.relative_position_bias_table,
            self.relative_position_index,
            axis=0,
        )
        return tf.transpose(relative_position_bias, [2, 0, 1])

    def call(
        self, x, mask=None, return_attns=False
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (B_, N, 3, self.num_heads, -1))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))

        q, k, v = tf.unstack(qkv, 3)

        scale = tf.cast(self.scale, dtype=qkv.dtype)
        q = q * scale
        attn = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))
        attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            num_win = tf.shape(mask)[0]
            attn = tf.reshape(
                attn, (B_ // num_win, num_win, self.num_heads, N, N)
            )
            attn = attn + tf.expand_dims(mask, 1)[None, ...]

            attn = tf.reshape(attn, (-1, self.num_heads, N, N))
            attn = tf.nn.softmax(attn, -1)
        else:
            attn = tf.nn.softmax(attn, -1)

        attn = self.attn_drop(attn)

        x = tf.matmul(attn, v)
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (B_, N, C))

        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attns:
            return x, attn
        else:
            return x


class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)


class SwinTransformerBlock(keras.Model):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer.  Default: layers.LayerNormalization
    """

    def __init__(self, dim, input_resolution, num_heads=4, head_dim=None,
                 window_size=7, shift_size=0, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=LayerNormalization, **kwargs,):
        super(SwinTransformerBlock, self).__init__(**kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(epsilon=1e-5)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="window_attention",
        )

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else tf.identity
        )
        self.norm2 = norm_layer(epsilon=1e-5)
       # self.mlp = mlp_block(
        #    dropout_rate=drop, hidden_units=[int(dim * mlp_ratio), dim]
        #)

        self.mlp = MLP(
            in_features=dim,
            out_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop
        )

        if self.shift_size > 0:
            # `get_attn_mask()` uses NumPy to make in-place assignments.
            # Since this is done during initialization, it's okay.
            self.attn_mask = self.get_attn_mask()
        else:
            self.attn_mask = None

    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        H, W = self.input_resolution
        img_mask = np.zeros((1, H, W, 1))  # [1, H, W, 1]
        cnt = 0
        for h in (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        ):
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                img_mask[:, h, w, :] = cnt
                cnt += 1

        img_mask = tf.convert_to_tensor(img_mask, dtype="float32")
        mask_windows = window_partition(
            img_mask, self.window_size
        )  # [num_win, window_size, window_size, 1]
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
            mask_windows, 2
        )
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        return tf.where(attn_mask == 0, 0.0, attn_mask)

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]

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
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA

        attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = window_reverse(
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

        return x


class PatchMerging(keras.layers.Layer):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=LayerNormalization, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False, name='downsample_reduction')
        self.norm = LayerNormalization(epsilon=1e-5)

    def call(self, x):
        """
        x: B, H * W, C
        """
        H, W = self.input_resolution
        B, L, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)

        x = tf.reshape(x, (B, -1, 4 * C))

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchEmbed(keras.layers.Layer):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
        self.img_size = (img_size)
        self.patch_size = (patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(filters=embed_dim,
                           kernel_size=patch_size,
                           strides=patch_size,
                           name="proj"
                          )
        if norm_layer is not None:
            self.norm = norm_layer(epsilon=1e-5)
        else:
            self.norm = None

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
      #  assert H == self.img_size[0] and W == self.img_size[1], \
       #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x)
        x = tf.reshape(x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim])
        if self.norm is not None:
            x = self.norm(x)
        return x


class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer. Default: layers.LayerNormalization
        downsample (layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, out_dim, input_resolution, depth, num_heads=4,
                 head_dim=None, window_size=7, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0,
                 norm_layer=LayerNormalization, downsample=None, **kwargs):

        super(BasicLayer, self).__init__(kwargs)

        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # build blocks
        blocks = [
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                name=f"swin_transformer_block_{i}",
            )
            for i in range(depth)
        ]
        self.blocks = blocks

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution,
                dim=dim,
             #   norm_layer=norm_layer,
                name="downsample"
                )
        else:
            self.downsample = None

    def call(self, x) :

        for i, block in enumerate(self.blocks):
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)

        return x


class SwinTransformer(keras.Model):
    """Swin Transformer
        A TensorFlow impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        head_dim (int, tuple(int)):
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (layers.Layer): Normalization layer. Default: layers.LayerNormalization.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        pre_logits (bool): If True, return model without classification head. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, num_classes=1000,
                 global_pool="avg", embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), head_dim=None, window_size=7,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=LayerNormalization,
                 ape=False, patch_norm=True, in_channels=3, include_top=True, **kwargs,):

        super(SwinTransformer, self).__init__(**kwargs)

        self.img_size = (
            img_size
            if isinstance(img_size, collections.abc.Iterable)
            else (img_size, img_size)
        )
        self.patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.ape = ape
        self.patch_norm = patch_norm
        self.in_channels = in_channels
        self.include_top = include_top

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            name="patch_embedding"
        )

        self.patch_grid = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = tf.Variable(
                tf.zeros((1, self.num_patches, self.embed_dim)),
                trainable=True,
                name="absolute_pos_embed",
            )
        else:
            self.absolute_pos_embed = None
        self.pos_drop = Dropout(drop_rate)

        # build layers
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = [
                int(self.embed_dim * 2 ** i) for i in range(self.num_layers)
            ]
        embed_out_dim = self.embed_dim[1:] + [None]
        head_dim = to_ntuple(self.num_layers)(head_dim)
        window_size = to_ntuple(self.num_layers)(window_size)
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [
            float(x) for x in tf.linspace(0.0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        layers = [
            BasicLayer(
                dim=self.embed_dim[i],
                out_dim=embed_out_dim[i],
                input_resolution=(
                    self.patch_grid[0] // (2 ** i),
                    self.patch_grid[1] // (2 ** i),
                ),
                depth=depths[i],
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
               # downsample=PatchMerging if (i < self.num_layers - 1) else None,
                downsample=PatchMerging if (i < self.num_layers - 1) else None,
                name=f"layer{i}",
            )
            for i in range(self.num_layers)
        ]
        self.swin_layers = layers
        self.norm = norm_layer(epsilon=1e-5)

        if self.include_top:
            self.head = Dense(num_classes, name="classification_head")

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for swin_layer in self.swin_layers:
            x = swin_layer(x)

        x = self.norm(x)  # [B, L, C]
        return x

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = tf.reduce_mean(x, axis=1)
        return x if not self.include_top else self.head(x)

    def call(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
