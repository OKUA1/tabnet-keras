from typing import Dict, List, Tuple, Optional, Union, Callable
import tensorflow as tf
from tabnet_keras.encoder import TabNetEncoder


class TabNetClassifier(tf.keras.Model):

    def __init__(
        self,
        n_classes: int,
        out_activation: Optional[Callable] = None,
        decision_dim: int = 8,
        attention_dim: int = 8,
        n_steps: int = 3,
        n_shared_glus: int = 2,
        n_dependent_glus: int = 2,
        relaxation_factor: float = 1.3,
        epsilon: float = 1e-15,
        virtual_batch_size: Optional[int] = None,  # not used
        momentum: float = 0.98,
        mask_type: str = "sparsemax",
        lambda_sparse: float = 1e-3,
        **kwargs
    ):
        """
        Creates a TabNet encoder network.
        Parameters:
        -----------
        n_classes: int
            Number of classes
        out_activation: Optional[Callable]
            Activation function to be used in the output layer. 
            Returns logits if None.
        decision_dim: int
            Dimension of the decision layer. Typically ranges from 8 to 128. Assuming 
            decision_dim to be equal to attention_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        attention_dim: int
            Dimension of the attention layer. Typically ranges from 8 to 128. Assuming 
            attention_dim to be equal to decision_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the 
            data has more informative features, the number of steps is higher. Large 
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the 
            number of shared layers is an effective strategy to improve predictive performance 
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        relaxation_factor: float
            Relaxation parameter used to compute the prior in the Attentive Transformer 
            layers. Typically ranges from 1.0 to 2.0. This is an important hyperparameter 
            to tune in TabNets. Default (1.3).
        epsilon: float
            Prevent computing log(0) by adding a small constant log(0+epsilon). Default (1e-15).
        virtual_batch_size: int
            Not used for not due to incorrect behaviour
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Default(sparsemax) 
            generates a highly sparse mask. 
            To learn more, refer: https://arxiv.org/abs/1905.05702.
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values 
            lead to better performance. Default (1e-3).
        """
        super(TabNetClassifier, self).__init__(**kwargs)

        self.tabnet_encoder = TabNetEncoder(decision_dim = decision_dim,
                                    attention_dim = attention_dim,
                                    n_steps = n_steps,
                                    n_shared_glus = n_shared_glus,
                                    n_dependent_glus = n_dependent_glus,
                                    relaxation_factor = relaxation_factor,
                                    epsilon = epsilon,
                                    virtual_batch_size = virtual_batch_size,
                                    momentum = momentum,
                                    mask_type = mask_type,
                                    lambda_sparse = lambda_sparse,
                                    **kwargs)

        
        self.clf = tf.keras.layers.Dense(
            n_classes, activation=out_activation, use_bias=False, name='classifier')

    def call(self, inputs, training=None):
        self.activations = self.tabnet_encoder(inputs, training=training)
        out = self.clf(self.activations)
        return out


class TabNetRegressor(tf.keras.Model):

    def __init__(
        self,
        n_regressors: int = 1,
        decision_dim: int = 8,
        attention_dim: int = 8,
        n_steps: int = 3,
        n_shared_glus: int = 2,
        n_dependent_glus: int = 2,
        relaxation_factor: float = 1.3,
        epsilon: float = 1e-15,
        virtual_batch_size: Optional[int] = None,  # not used
        momentum: float = 0.98,
        mask_type: str = "sparsemax",
        lambda_sparse: float = 1e-3,
        **kwargs
    ):
        """
        Creates a TabNet encoder network.
        Parameters:
        -----------
        n_classes: int
            Number of regressors
        decision_dim: int
            Dimension of the decision layer. Typically ranges from 8 to 128. Assuming 
            decision_dim to be equal to attention_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        attention_dim: int
            Dimension of the attention layer. Typically ranges from 8 to 128. Assuming 
            attention_dim to be equal to decision_dim is sensible. Large values may lead 
            to overfitting. Default (8).
        n_steps: int
            Number of sequential attention steps. Typically ranges from 3 to 10. If the 
            data has more informative features, the number of steps is higher. Large 
            values may lead to overfitting. Default (3).
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Increasing the 
            number of shared layers is an effective strategy to improve predictive performance 
            without a significant increase in the number of parameters. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Increasing 
            the number of step-dependent layers is an effective strategy to improve predictive 
            performance. Default (2).
        relaxation_factor: float
            Relaxation parameter used to compute the prior in the Attentive Transformer 
            layers. Typically ranges from 1.0 to 2.0. This is an important hyperparameter 
            to tune in TabNets. Default (1.3).
        epsilon: float
            Prevent computing log(0) by adding a small constant log(0+epsilon). Default (1e-15).
        virtual_batch_size: int
            Not used for not due to incorrect behaviour
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values 
            correspond to larger impact of batch on the rolling statistics computed in 
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        mask_type: str
            mask_type ∈ {"softmax", "entmax", "sparsemax"}. Softmax generates a dense mask.
            Entmax (i.e. entmax 1.5) generates a slightly sparser mask. Default(sparsemax) 
            generates a highly sparse mask. 
            To learn more, refer: https://arxiv.org/abs/1905.05702.
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values 
            lead to better performance. Default (1e-3).
        """
        super(TabNetRegressor, self).__init__(**kwargs)

        self.tabnet_encoder = TabNetEncoder(decision_dim = decision_dim,
                                    attention_dim = attention_dim,
                                    n_steps = n_steps,
                                    n_shared_glus = n_shared_glus,
                                    n_dependent_glus = n_dependent_glus,
                                    relaxation_factor = relaxation_factor,
                                    epsilon = epsilon,
                                    virtual_batch_size = virtual_batch_size,
                                    momentum = momentum,
                                    mask_type = mask_type,
                                    lambda_sparse = lambda_sparse,
                                    **kwargs)

        
        self.clf = tf.keras.layers.Dense(
            n_regressors, use_bias=False, name='regressor')

    def call(self, inputs, training=None):
        self.activations = self.tabnet_encoder(inputs, training=training)
        out = self.clf(self.activations)
        return out
