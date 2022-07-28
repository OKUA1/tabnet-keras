from typing import Dict, List, Tuple, Optional, Union
import tensorflow as tf

from tabnet_keras.split import Split
from tabnet_keras.attentive_transformer import AttentiveTransformer
from tabnet_keras.feature_transformer import FeatureTransformer

class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(
        self, 
        decision_dim: int = 8, 
        attention_dim: int = 8, 
        n_steps: int = 3, 
        n_shared_glus: int = 2, 
        n_dependent_glus: int = 2, 
        relaxation_factor: float = 1.3, 
        epsilon: float = 1e-15, 
        virtual_batch_size: Optional[int] = None, 
        momentum: float = 0.98, 
        mask_type: str = "sparsemax", 
        lambda_sparse: float = 1e-3, 
        **kwargs
    ):
        """
        Creates a TabNet encoder network.
        Parameters:
        -----------
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
        lambda_sparse: float
            Coefficient for the mask sparsity loss. Important parameter to tune. Lower values 
            lead to better performance. Default (1e-3).
        """
        super(TabNetEncoder, self).__init__(**kwargs)
        self.decision_dim = decision_dim
        self.n_steps = n_steps
        self.n_dependent_glus = n_dependent_glus
        self.relaxation_factor = relaxation_factor
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.lambda_sparse = lambda_sparse

        # plain batch normalization
        self.initial_bn = tf.keras.layers.BatchNormalization(momentum=self.momentum)

        # shared glu layers
        self.glu_dim = decision_dim + attention_dim
        self.shared_glu_fc_layers = list()
        for _ in range(n_shared_glus):
            self.shared_glu_fc_layers.append(tf.keras.layers.Dense(units=self.glu_dim*2, use_bias=False))
        
        # initial feature transformer
        self.initial_feature_transformer = FeatureTransformer(
            n_dependent_glus=self.n_dependent_glus, 
            shared_glu_fc_layers=self.shared_glu_fc_layers, 
            units=self.glu_dim, 
            virtual_batch_size=self.virtual_batch_size, 
            momentum=self.momentum, 
            name="FeatureTransformer_Step_0", 
        )

        # split layer
        self.split_layer = Split(split_dim=decision_dim)
    
    def build(self, input_shape: tf.TensorShape):
        feature_dim = input_shape[-1]

        # feature and attentive transformers for each step
        self.step_feature_transformers = list()
        self.step_attentive_transformers = list()
        for step in range(self.n_steps):
            feature_transformer = FeatureTransformer(
                n_dependent_glus=self.n_dependent_glus, 
                shared_glu_fc_layers=self.shared_glu_fc_layers, 
                units=self.glu_dim, 
                virtual_batch_size=self.virtual_batch_size, 
                momentum=self.momentum, 
                name=f"FeatureTransformer_Step_{(step+1)}", 
            )
            attentive_transformer = AttentiveTransformer(
                units=feature_dim, 
                n_steps=self.n_steps, 
                epsilon=self.epsilon, 
                lambda_sparse=self.lambda_sparse, 
                virtual_batch_size=self.virtual_batch_size, 
                momentum=self.momentum, 
                mask_type = self.mask_type, 
                name=f"AttentiveTransformer_Step_{(step+1)}", 
            )
            self.step_feature_transformers.append(
                feature_transformer
            )
            self.step_attentive_transformers.append(
                attentive_transformer
            )

    def call(self, inputs: tf.Tensor, prior: Optional[tf.Tensor] = None, 
             training: Optional[bool] = None) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]
        step_output_aggregate = tf.zeros((batch_size, self.decision_dim))
        
        if prior is None:
            prior = tf.ones_like(inputs)
        
        x = self.initial_bn(inputs, training=training)
        x_proc = self.initial_feature_transformer(x, training=training)
        _, x_a = self.split_layer(x_proc)

        for step in range(self.n_steps):
            # step operations
            mask = self.step_attentive_transformers[step](x_a, 
                                                          prior=prior, 
                                                          training=training)
            masked_x = tf.multiply(mask, x)
            x_proc = self.step_feature_transformers[step](masked_x, 
                                                          training=training)
            x_d, x_a = self.split_layer(x_proc)
            step_output = tf.keras.activations.relu(x_d)

            # for prediction
            step_output_aggregate += step_output

            # update prior
            prior = tf.multiply(self.relaxation_factor - mask, prior)

        return step_output_aggregate
    
    def calculate_feature_attribution(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, Dict[int, tf.Tensor]]:
        feature_attribution = tf.zeros_like(inputs)
        masks = dict()
        prior = tf.ones_like(inputs)
        
        x = self.initial_bn(inputs, training=False)
        x_proc = self.initial_feature_transformer(x, training=False)
        _, x_a = self.split_layer(x_proc)

        for step in range(self.n_steps):
            # step operations
            mask = self.step_attentive_transformers[step](x_a, 
                                                          prior=prior, 
                                                          training=False)
            masked_x = tf.multiply(mask, x)
            x_proc = self.step_feature_transformers[step](masked_x, 
                                                          training=False)
            x_d, x_a = self.split_layer(x_proc)
            step_output = tf.keras.activations.relu(x_d)

            # for interpretation
            masks[step] = mask
            step_coefficient = tf.reshape(
                tf.math.reduce_sum(step_output, axis=-1), 
                shape=(-1,1)
            )
            feature_attribution += tf.multiply(step_coefficient, mask)
            
            # update prior
            prior = tf.multiply(self.relaxation_factor - mask, prior)
        
        # normalization
        feature_attribution /= tf.reshape(
            tf.reduce_sum(feature_attribution, axis=-1), 
            shape=(-1,1)
        )
        
        return feature_attribution, masks