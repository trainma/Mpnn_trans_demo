from fbprophet import Prophet
from statsmodels.tsa.arima_model import ARIMA
import numpy as np
import scipy.sparse as sp


def arima(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error = np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp, n_samples - ahead):  #
        print(test_sample)
        count += 1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j, :test_sample - 1].reset_index()

            if (sum(ds.iloc[:, 1]) == 0):
                yhat = [0] * (ahead)
            else:
                try:
                    fit2 = ARIMA(ds.iloc[:, 1].values, (2, 0, 2)).fit()
                except:
                    fit2 = ARIMA(ds.iloc[:, 1].values, (1, 0, 0)).fit()
                # yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
                yhat = abs(fit2.predict(start=test_sample, end=(test_sample + ahead - 2)))
            y_me = labels.iloc[j, test_sample:test_sample + ahead]
            e = abs(yhat - y_me.values)
            err += e
            error += e

        for idx in range(ahead):
            var[idx].append(err[idx])
    return error, var


def prophet(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error = np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp, n_samples - ahead):  #
        print(test_sample)
        count += 1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j, :test_sample].reset_index()
            ds.columns = ["ds", "y"]
            # with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            y_me = labels.iloc[j, test_sample:test_sample + ahead]
            e = abs(yhat - y_me.values).values
            err += e
            error += e
        for idx in range(ahead):
            var[idx].append(err[idx])

    return error, var
