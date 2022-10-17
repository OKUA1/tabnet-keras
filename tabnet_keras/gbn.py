import tensorflow as tf

class GhostBatch1DNormalization(tf.keras.Model):
    def __init__(
        self, virtual_batch_splits: int = 1, momentum: float = 0.9
    ):
        super(GhostBatch1DNormalization, self).__init__()
        self.virtual_batch_splits = virtual_batch_splits
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)

    def call(self, x, training = None):
        if training:
            full_div_size = (tf.shape(x)[0] // self.virtual_batch_splits) * self.virtual_batch_splits
            if full_div_size > 1:
                chunks = tf.split(x[0:full_div_size,:], self.virtual_batch_splits, axis = 0)
                x_ = [self.bn(c, training=True) for c in chunks]
                x_.append(self.bn(x[full_div_size:, :], training=False))
                concat = tf.concat(x_, 0)
                return concat
            else: 
                return self.bn(x, training = True)
        else: 
            return self.bn(x, training = False)