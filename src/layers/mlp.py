class MLP(keras.Model):
    """
        this class is a implementation of the mlp block described in the swin transformer paper, which contains 
        2 fully connected layer with GelU activation.
    """
    def __init__(self, input_dims=None, hidden_neurons=None,
                 output_neurons=None, act_type="gelu", dropout_rate=0., prefix=''):
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
        
        self.input_dims = input_dims
        self.fc_1 = Dense(units=hidden_neurons, name=f"{prefix}/mlp/dense_1")
        if act_type == "relu":
            self.act = tf.keras.layers.ReLU()
        else:
            self.act = None
        self.fc_2 = Dense(units=hidden_neurons, name=f"{prefix}/mlp/dense_2")
        self.drop = Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        if not self.act:
            x = tf.keras.activations.gelu(x)
        else: 
            x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
    def summary(self): 
        inputs = Input(shape=self.input_dims)
        return keras.Model(inputs=inputs, outputs=self.call(inputs))
