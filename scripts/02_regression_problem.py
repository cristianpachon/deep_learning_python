from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

train_data.shape
test_data.shape
train_targets

# Standardisation
mean = train_data.mean(axis = 0)
std = train_data.std(axis = 0)

train_data -= mean
train_data /= std

test_data -= mean
test_data /= std

# Defining the model
def build_model():
    model = models.Sequential()

    model.add(
        layers.Dense(
            64,
            activation = 'relu',
            input_shape = (train_data.shape[1], )
        )
    )
    model.add(
        layers.Dense(
            64,
            activation = 'relu'
        )
    )

    model.add(
        layers.Dense(1)
    )

    model.compile(
        optimizer = 'rmsprop',
        loss = 'mse',
        metrics = ['mae']
    )

k = 4
num_val_samples = len(train_data)// k
num_epochs = 100
all_scores = []
