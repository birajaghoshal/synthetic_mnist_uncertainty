import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, Lambda
from tensorflow.keras import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist


def define_model_a():
    model = Sequential([
        Lambda(lambda x: x / 255.),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(.2),
        Dense(512, activation='relu'),
        Dropout(.2),
        Dense(10, activation='softmax')
    ])
    optim = tf.keras.optimizers.Adam(1e-3)
    model.compile(opitimizer=optim, metrics=['accuracy'], loss=SparseCategoricalCrossentropy())
    return model


def define_model_b():
    model = Sequential([
        Lambda(lambda x: x / 255.),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(.2),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    optim = tf.keras.optimizers.Adam(1e-3)
    model.compile(opitimizer=optim, metrics=['accuracy'], loss=SparseCategoricalCrossentropy())
    return model


def define_model_c():
    model = Sequential([
        Lambda(lambda x: x / 255.),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dropout(.4),
        Dense(10, activation='softmax')
    ])
    optim = tf.keras.optimizers.Adam(1e-3)
    model.compile(opitimizer=optim, metrics=['accuracy'], loss=SparseCategoricalCrossentropy())
    return model


def train(x, y, model, epochs=10):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    model.fit(x, y,
              batch_size=32,
              epochs=epochs,
              verbose=2)
    return model


def predict(model, x, samples=20):
    '''
    Make a prediction using MC Dropout
    Args:
        model: The trained keras model
        x: the input tensor with shape [N, M]
        samples: the number of monte carlo samples to collect
    Returns:
        probs: The expected value of our prediction
        entropy: The standard deviation of our prediction
    '''
    yhat_arr = []

    for t in range(samples):
        yhat = model(x, training=True)
        yhat_arr.append(yhat)

    yhat_arr = np.stack(yhat_arr, -1)
    probs = np.mean(yhat_arr, axis=-1)
    entropy = - 1.0 * np.sum(probs * np.log(probs + 1e-16), axis=-1)
    return probs, entropy


model_builders = [define_model_a, define_model_b, define_model_c]

if __name__ == '__main__':
    (x_trn, y_trn), (x_tst, y_tst) = mnist.load_data()

    for i, build in enumerate(model_builders):
        print("training model-%d.h5" % i)
        model = build()
        model = train(x_trn, y_trn, model)
        model.save("model-%d.h5" % i)
