# TF2 Keras implementation of TabNet

TabNet is a novel deep learning architecture for **tabular data**. TabNet performs reasoning in multiple decision steps and using sequential attention to select which features to use at which decision step. You can find more information about it in the original [research paper](https://arxiv.org/abs/1908.07442).

## Installation 

```bash
$ pip install tabnet_keras
```

## Usage

```python
from tabnet_keras import TabNetRegressor, TabNetClassifier

tabnet_params = {
    "decision_dim": 16,
    "attention_dim": 16,
    "n_steps": 3,
    "n_shared_glus": 2,
    "n_dependent_glus": 2,
    "relaxation_factor": 1.3,
    "epsilon": 1e-15,
    "momentum": 0.98,
    "mask_type": "sparsemax", # can be 'sparsemax' or 'softmax'
    "lambda_sparse": 1e-3, 
    "virtual_batch_splits": 8 #number of splits for ghost batch normalization, ideally should evenly divide the batch_size
}

### Regression 
model = TabNetRegressor(n_regressors = 1, **tabnet_params)
model.compile(loss = 'mean_squared_error', optimizer = tf.keras.optimizers.Adam(0.01), 
             metrics = [tf.keras.metrics.RootMeanSquaredError()])
model.fit(X, y, epochs = 100, batch_size = 1024)

### Classification
model = TabNetClassifier(n_classes = 10, out_activation = None, **tabnet_params)
model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(0.01))
model.fit(X, y, epochs = 100, batch_size = 1024)

```

## Acknowledgment

Most of the code is taken with minor changes from [this repository](https://github.com/jeyabbalas/tabnet).
