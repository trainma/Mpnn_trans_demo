import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('./')


def plot(pred, true):
    fig = plt.figure(figsize=(20, 8), dpi=1000)
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred, label="pred")
    ax.plot(range(len(true)), true, label="ground truth")
    plt.legend()
    plt.xlabel("Hours")
    plt.ylabel("PM2.5 Values")
    plt.show()


def RMSE(pred, true):
    return np.sqrt(np.mean(np.square(pred - true)))


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean(np.square(pred - true))


def plot_history(history, figsize, dpi=200):
    fig = plt.figure(dpi=dpi, figsize=figsize)
    plt.title('Train History')
    ax = plt.subplot(2, 1, 1)
    ax.plot(history.history['mae'])
    ax.plot(history.history['val_mae'])
    plt.title('Model Mae')
    plt.ylabel('mae')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    # 绘制训练 & 验证的损失值
    ax = plt.subplot(2, 1, 2)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.show()


def plot_prediction_true(pred, true):
    fig = plt.figure(figsize=(24, 8), dpi=1000)
    ax = fig.add_subplot(111)
    ax.plot(range(len(pred)), pred, label="Prediction")
    ax.plot(range(len(true)), true, label="Ground truth")
    plt.legend()
    plt.xlabel("Time(hourly)")
    plt.ylabel("PM2.5 values")
    plt.show()
