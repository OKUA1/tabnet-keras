from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf

class Split(tf.keras.layers.Layer):
    def __init__(
        self, 
        split_dim: int = 8, 
        **kwargs
    ):
        """
        Splits the input tensor into two at a specified column dimension.
        Parameters:
        -----------
        split_dim: int
            Column dimension where the input tensor should be split into two. Default (8).
        """
        super(Split, self).__init__(**kwargs)
        self.split_dim = split_dim
    
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor]:
        return inputs[:, :self.split_dim], inputs[:, self.split_dim:]