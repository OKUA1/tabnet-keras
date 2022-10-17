from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tabnet_keras.glu import GLULayer

class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(
            self, 
            n_dependent_glus: int = 2, 
            shared_glu_fc_layers: Optional[List[tf.keras.layers.Dense]] = None, 
            units: int = 16, 
            virtual_batch_splits: Optional[int] = None, 
            momentum: float = 0.98, 
            **kwargs
    ):
        """
        Creates a feature transformer for non-linear processing of features.
        Parameters:
        -----------
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        shared_glu_fc_layers: List[tf.keras.layers.Dense]
            A list of dense layers to construct shared GLU layers. Default (None) creates only 
            n_dependent_glus dependent GLU layers and no shared layers. Total number of GLU layers 
            in this feature transformer is len(shared_glu_layers) + n_dependent_glus.
        units: int
            Number of units in each GLU layer. Default (16).
        virtual_batch_splits: int
            Number of splits for ghost batch normalization. Preferrably should divide the batch size. Otherwise, 
            the last virtual batch is not used for batch_norm training.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(FeatureTransformer, self).__init__(**kwargs)
        n_glu_layers = (len(shared_glu_fc_layers) if shared_glu_fc_layers else 0) + n_dependent_glus
        if n_glu_layers <= 0:
            raise ValueError("Invalid Argument: Total number of GLU layers in the feature transformer"
            " should be greater than 0.")
        
        self.units = units
        self.norm_factor = tf.math.sqrt(tf.constant(0.5))

        self.glu_layers = list()
        for i in range(n_glu_layers):
            fc_layer = None
            if shared_glu_fc_layers:
                if i < len(shared_glu_fc_layers):
                    fc_layer = shared_glu_fc_layers[i]
            
            glu_layer = GLULayer(
                units=self.units, 
                fc_layer=fc_layer, 
                virtual_batch_splits=virtual_batch_splits, 
                momentum=momentum, 
            )
            self.glu_layers.append(glu_layer)
    
    def build(self, input_shape: tf.TensorShape):
        if input_shape[-1] != self.units:
            self.omit_first_residual = True
        else: 
            self.omit_first_residual = False
    
    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        for i, glu_layer in enumerate(self.glu_layers):
            x = glu_layer(inputs, training=training)
            if self.omit_first_residual and (i==0):
                inputs = x
            else:
                x = tf.math.multiply(self.norm_factor, tf.math.add(inputs, x))
                inputs = x

        return x