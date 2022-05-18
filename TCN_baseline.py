'''
Implementing the TCN baseline method in Keras and Tensorflow API with custom dataset.
2022/05/18 xhz.

'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from tcn.tcn import TCN
from tensorflow import keras
from tqdm import tqdm

import utils.metrics as metrics
from utils import visualization as vz


def get_dataset(Source, scope, train_test_split, window_size, n_out):
    split = int(train_test_split * scope)
    df = Source
    scaler = MinMaxScaler()
    scaler2 = MinMaxScaler()
    open_arr = scaler.fit_transform(df.values)
    labels = scaler2.fit_transform(df['no2'].values.reshape(-1, 1))
    X = np.zeros(shape=(scope - window_size, window_size, open_arr.shape[1]))
    label = np.zeros(shape=(scope - window_size, 1))
    for i in range(scope - window_size):
        X[i, :] = open_arr[i:i + window_size, :]
        label[i, :] = labels[i + window_size, 0]
    train_X = X[:split, :]
    train_label = label[:split]
    test_X = X[split:scope, :]
    test_label = label[split:scope]
    return train_X, train_label, test_X, test_label, scaler, scaler2


def get_dataset_multi_step(Source, scope, train_test_split, window_size, n_out):
    split = int(train_test_split * scope)
    df = Source
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df.values)
    label_list = np.zeros(shape=(scope - window_size - n_out, n_out, df.shape[-1]))

    for i in range(scope - window_size - n_out):
        label_list[i, :] = data[i + window_size:i + window_size + n_out, :]

    X = np.zeros(shape=(scope - window_size - n_out, window_size, data.shape[-1]))

    for i in range(scope - window_size - n_out):
        X[i, :] = data[i:i + window_size, :]

    train_X = X[:split, :]
    train_label = label_list[:split, :]
    test_X = X[split:scope, :]
    test_label = label_list[split:scope, :]
    return train_X, train_label, test_X, test_label, scaler


class metric():
    def __init__(self, pred, true):
        self.pred = pred
        self.true = true

    def RMSE(self):
        return np.sqrt(np.mean(np.square(self.pred - self.true)))

    def plot(self):
        fig = plt.figure(figsize=(24, 8), dpi=1000)
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.pred)), self.pred, label="Prediction")
        ax.plot(range(len(self.true)), self.true, label="Ground truth")
        plt.legend()
        plt.xlabel("Time(hourly)")
        plt.ylabel("PM2.5 values")
        plt.show()

    def MAE(self):
        return np.mean(np.abs(self.pred - self.true))

    def MSE(self):
        return np.mean(np.square(self.pred - self.true))

    def R2(self):
        SST = np.sum(np.power((self.true - np.mean(self.true)), 2))
        SSReg = np.sum(np.power((self.pred - np.mean(self.true)), 2))
        return SSReg / SST


# %%
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


if __name__ == '__main__':
    window_size = 72  # ths size of window
    batch_size = 32
    epochs = 100
    filter_nums = 64
    kernel_size = 4
    pred_len = 24
    df = pd.read_csv('./new.csv')
    df = df.drop(['Unnamed: 0', 'Time'], axis=1)
    feature_size = df.shape[1]
    Savepath = './new'
    train_X, train_label, test_X, test_label, scaler = get_dataset_multi_step(df, 35000, 0.9,
                                                                              window_size=window_size,
                                                                              n_out=pred_len)

    inputs = keras.layers.Input(shape=(window_size, feature_size))  # Input [batch_size, window_size, feature_size]
    model = keras.models.Sequential([
        keras.layers.Input(shape=(window_size, feature_size)),
        TCN(nb_filters=32,  # filter number like units
            kernel_size=2,  # Kernal number
            dilations=[1, 2, 4, 8, 16],
            ),  # 空洞因子

        keras.layers.Dense(units=feature_size * 2 * pred_len, activation='relu'),
        keras.layers.Dense(units=feature_size * pred_len),
        keras.layers.Reshape((pred_len, 12), )  # Outshape: [pred_len,feature_size]

    ])
    model.summary()
    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5, decay=1e-4, clipnorm=0.2), loss='mae',
                  metrics=['mae'])

    filepath = Savepath + "weights-improvement-{epoch:02d}-{val_mae:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=100, save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]
    history = model.fit(train_X, train_label, validation_split=0.1,
                        epochs=epochs, callbacks=callbacks_list, batch_size=64)
    vz.plot_history(history=history, figsize=(20, 10))
    evaluate_res = model.evaluate(test_X, test_label)
    prediction = model.predict(test_X)
    mae_list = []
    rmse_list = []
    for i in tqdm(range(prediction.shape[0])):
        pred_temp = prediction[i, :, :]
        real_temp = test_label[i, :, :]
        inverse_pred = scaler.inverse_transform(pred_temp)
        inverse_real = scaler.inverse_transform(real_temp)
        mae, mse, rmse, mape, mspe = metrics.metric(inverse_pred, inverse_real)
        mae_list.append(mae)
        rmse_list.append(rmse)
    print("Test MAE:", sum(mae_list) / len(mae_list))
    print("Test RMSE:", sum(rmse_list) / len(rmse_list))
