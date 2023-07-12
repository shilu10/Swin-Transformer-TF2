from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from tensorflow.keras import Model 
from tensorflow.keras.layers import Layer 
from tensorflow.keras.layers import *


class MLP(tf.keras.layers.Layer):
    """
        this class is a implementation of the mlp block described in the swin transformer paper, which contains 
        2 fully connected layer with GelU activation.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        """
            Params:
                input_neurons(dtype: int)   : input dimension for the mlp block, it needed only for .summary() method.
                input_dims(dtype: int)  : number of neurons in the hidden
                                              layer(fully connected layer).
                output_neurons(dtype: iny)  ; number of neurons in the last
                                              layer(fully connected layer) of mlp.
                act_type(type: str)         ; type of activation needed. in paper, GeLU is used.
                dropout_rate(dtype: float)  : dropout rate in the dropout layer.
                prefix(type: str)           : used for the naming the layers.
        """
        super(MLP, self).__init__()
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

    def summary(self): 
        inputs = Input(shape=self.input_dims)
        return keras.Model(inputs=inputs, outputs=self.call(inputs))
