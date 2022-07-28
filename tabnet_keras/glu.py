from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf

class GLULayer(tf.keras.layers.Layer):
    def __init__(
            self, 
            units: int = 16, 
            fc_layer: Optional[tf.keras.layers.Dense] = None,
            momentum: float = 0.98,
            virtual_batch_size = None, #not used due to incorrect behaviour 
            **kwargs
    ):
        """
        Creates a layer with a fully-connected linear layer, followed by batch 
        normalization, and a gated linear unit (GLU) as the activation function.
        Parameters:
        -----------
        units: int
            Number of units in layer. Default (16).
        fc_layer:tf.keras.layers.Dense
            This is useful when you want to create a GLU layer with shared parameters. This 
            is necessary because batch normalization should still be uniquely initialized 
            due to the masked inputs in TabNet steps being in a different scale than the 
            original input. Default (None) creates a new FC layer.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(GLULayer, self).__init__(**kwargs)
        self.units = units
        self.momentum = momentum

        if fc_layer:
            self.fc = fc_layer
        else:
            self.fc = tf.keras.layers.Dense(self.units*2, use_bias=False)
        
        self.bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.math.multiply(x[:, :self.units], tf.nn.sigmoid(x[:, self.units:]))
        return x