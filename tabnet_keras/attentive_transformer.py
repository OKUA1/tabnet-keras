from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf
from tabnet_keras.activations.sparsemax import sparsemax

class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(
        self, 
        units: int, 
        n_steps: int = 3, 
        epsilon: float = 1e-15, 
        lambda_sparse: float = 1e-3, 
        virtual_batch_size: Optional[int] = None,  #not used due to incorrect behaviour 
        momentum: float = 0.98, 
        mask_type: str = "sparsemax", 
        **kwargs
    ):
        """
        Creates an attentive transformer that learns masks to select salient features 
        for further analysis. This layer propagates the sparsity regularization loss 
        to the Keras model/layer that calls this layer.
        Parameters:
        -----------
        units: int
            Number of units in layer. This layer outputs a mask for your data, so the 
            number of units should be the same as your data dimension.
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the 
            data has more informative features, the number of steps is higher. Large 
            values may lead to overfitting. Default (3).
        epsilon: float
            Prevent computing log(0) by adding a small constant log(0+epsilon). Default (1e-15).
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values 
            lead to better performance. Default (1e-3).
        virtual_batch_size: int
            Batch size for Ghost Batch Normalization (GBN). Value should be much smaller 
            than and a factor of the overall batch size. Default (None) runs regular batch 
            normalization. If an integer value is specified, GBN is run with that virtual 
            batch size.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Default(sparsemax) 
            generates a highly sparse mask. 
            To learn more, refer: https://arxiv.org/abs/1905.05702.
        """
        super(AttentiveTransformer, self).__init__(**kwargs)
        # for computing sparsity regularization loss
        self.n_steps = n_steps
        self.epsilon = epsilon
        self.lambda_sparse = lambda_sparse
        
        # attentive transformer layers
        self.fc = tf.keras.layers.Dense(units, use_bias=False)

        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)

        if mask_type == "sparsemax":
            self.sparse_activation = sparsemax
        # elif mask_type == "entmax":
        #     self.sparse_activation = entmax15
        elif mask_type == "softmax":
            self.sparse_activation = tf.nn.softmax
        else:
            raise NotImplementedError(
                "Available options for mask_type: {'sparsemax', 'softmax'}"
            )
        
    def call(self, inputs: tf.Tensor, prior: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        x = tf.multiply(prior, x)
        x = self.sparse_activation(x, axis=-1)

        # propagate sparsity loss from current mask
        sparsity_reg_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.multiply(-x, tf.math.log(x+self.epsilon)), 
                axis=-1
            )
        )
        sparsity_reg_loss /= self.n_steps
        self.add_loss(self.lambda_sparse*sparsity_reg_loss)

        return x