import tensorflow as tf
from typing import Optional
from tabnet_keras.feature_transformer import FeatureTransformer


class TabNetDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        n_features: int,
        n_steps: int = 3,
        n_shared_glus: int = 1,
        n_dependent_glus: int = 1,
        virtual_batch_splits: Optional[int] = None,
        momentum: float = 0.98,
        **kwargs,
    ):
        """
        Creates a TabNet encoder network.
        Parameters:
        -----------
        n_features: int
            Number of features in the original dataset
        n_steps: int
            Number of sequential attention steps. Default (3)
        n_shared_glus: int
            Number of shared GLU layers within the Feature Transformer. Default (2).
        n_dependent_glus: int
            Number of step-dependent GLU layers within the Feature Transformer. Default (2).
        virtual_batch_splits: int
            Number of splits for ghost batch normalization. Preferrably should divide the batch size. Otherwise,
            the last virtual batch is not used for batch_norm training.
        momentum: float
            Momentum for exponential moving average in batch normalization. Lower values
            correspond to larger impact of batch on the rolling statistics computed in
            each batch. Valid values range from 0.0 to 1.0. Default (0.98).
        """
        super(TabNetDecoder, self).__init__(**kwargs)

        self.n_features = n_features
        self.n_steps = n_steps
        self.n_shared_glus = n_shared_glus
        self.n_dependent_glus = n_dependent_glus
        self.virtual_batch_splits = virtual_batch_splits
        self.momentum = momentum

    def build(self, input_shape):
        hidden_dim = input_shape[-1]

        self.shared_glu_fc = [
            tf.keras.layers.Dense(units=hidden_dim * 2, use_bias=False)
            for _ in range(self.n_shared_glus)
        ]

        self.ft = []
        self.fc = []

        for step in range(self.n_steps):
            self.ft.append(
                FeatureTransformer(
                    n_dependent_glus=self.n_dependent_glus,
                    shared_glu_fc_layers=self.shared_glu_fc,
                    units=hidden_dim,
                    virtual_batch_splits=self.virtual_batch_splits,
                    momentum=self.momentum,
                    name=f"FeatureTransformer_Decoder_Step_{(step+1)}",
                )
            )

            self.fc.append(
                tf.keras.layers.Dense(
                    units=self.n_features,
                    use_bias=False,
                    name=f"FC_Decoder_Step_{(step+1)}",
                )
            )


    def call(self, inputs):
        
        reconstruction = []

        for step in range(self.n_steps):
            x = self.ft[step](inputs)
            x = self.fc[step](x)
            reconstruction.append(x)
        
        return tf.reduce_sum(reconstruction, axis = 0)