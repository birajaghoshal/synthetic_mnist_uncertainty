import tensorflow as tf
import numpy as np
import argparse
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, help="path to model")
parser.add_argument("image", type=str, help="path to image")

def load_image(path):
    pim = Image.open(path)
    pim = pim.convert("L")
    pim = pim.resize([28, 28])
    nim = np.array(pim)

    # import matplotlib.pyplot as plt
    # plt.imshow(nim)
    # plt.show()
    return nim

def predict(model, x):
    '''
    Make a prediction using normal inference

    Args:
        model:
            The trained keras model
        x:
            The input tensor with shape either [N, H, W] or [H, W]
    Returns:
        probs:
            The predicted probabilities [N, K]
        preds:
            The predicted class [N]
        entropy:
            The standard deviation of our prediction

    '''
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    probs = model.predict(x)
    preds = np.argmax(probs, axis=-1)
    entropy = - 1.0 * np.sum(probs * np.log(probs + 1e-16), axis=-1)
    return probs, preds, entropy

def predict_mcdropout(model, x, samples=20):
    '''
    Make a prediction using MC Dropout
    Args:
        model:
            The trained keras model
        x:
            The input tensor with shape either [N, H, W] or [H, W]
        samples:
            The number of monte carlo samples to collect
    Returns:
        probs:
            The expected value of predicted probabilities [N, K]
        preds:
            The predicted class [N]
        entropy:
            The standard deviation of our prediction
    '''
    if len(x.shape) == 2:
        x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    yhat_arr = []
    for t in range(samples):
        yhat = model(x, training=True)
        yhat_arr.append(yhat)

    yhat_arr = np.stack(yhat_arr, -1)
    probs = np.mean(yhat_arr, axis=-1)
    preds = np.argmax(probs, axis=-1)
    entropy = - 1.0 * np.sum(probs * np.log(probs + 1e-16), axis=-1)
    return probs, preds, entropy

if __name__ == '__main__':
    args = parser.parse_args()
    model = load_model(args.model)
    image = load_image(args.image)
    probs, preds, entropy = predict(model, image)
    print("%s -> entropy=%s class=%s score=%.3f" % (args.image, entropy[0], preds[0], probs[0][preds[0]]))
