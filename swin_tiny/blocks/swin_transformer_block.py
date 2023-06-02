from tensorflow import keras 
import tensorflow as tf 
from tensorflow.keras.layers import * 
from tensorflow.keras import Model 
from ..layers import DropPath, WindowAttention
from utils import window_partition, window_reverse
from ..layers import MLP


class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=LayerNormalization):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        

        self.norm1 = norm_layer(epsilon=1e-5, name=f'norm1')
        self.attn = WindowAttention(
                        dim,
                        window_size=(self.window_size, self.window_size),
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        attn_drop=attn_drop,
                        proj_drop=drop
                    )
        self.drop_path = DropPath(
            drop_path if drop_path > 0. else 0.)
        self.norm2 = norm_layer(epsilon=1e-5, name=f'norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp = self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                       drop=drop)


    def build(self, input_shape):
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
                initial_value=attn_mask, trainable=False, name=f'attn_mask', dtype=tf.float64)
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = tf.reshape(
            x_windows, shape=[-1, self.window_size * self.window_size, C])
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = tf.reshape(
            attn_windows, shape=[-1, self.window_size, self.window_size, C])
        
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])
        
        # FFN
        x = shortcut + self.drop_path(x)
        temp_x = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = temp_x + self.drop_path(x)

        return x

