from keras import models
from keras import layers
from keras import regularizers

### Movie review classification

# Original model
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (10000, )))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Another version
model = models.Sequential()
model.add(layers.Dense(4, activation='relu', input_shape = (10000, )))
model.add(layers.Dense(4, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# Adding L2 weight regularization to the model
model = models.Sequential()

model.add(
    layers.Dense(
        16,
        kernel_regularizer = regularizers.l2(0.001),
        activation = 'relu',
        input_shape = (10000, )
    )
)

model.add(
    layers.Dense(
        16,
        kernel_regularizer = regularizers.l2(0.002),
        activation = 'relu'
    )
)

model.add(
    layers.Dense(
        1,
        activation = 'sigmoid'
    )
)