import pandas as pd
from fbprophet import Prophet

from utils.metrics import *


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
        for j in range(labels.shape[1]):
            ds = labels.iloc[:test_sample, [1, j + 2]]
            ds.columns = ["ds", "y"]
            # with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead))
            yhat = future["yhat"].tail(ahead)
            y_me = labels.iloc[test_sample:test_sample + ahead, j + 2]
            e = abs(yhat - y_me.values).values
            err += e
            error += e
        for idx in range(ahead):
            var[idx].append(err[idx])

    return error, var


if __name__ == "__main__":
    df = pd.read_csv('new.csv')
    error, var = prophet(ahead=1, start_exp=72, n_samples=5000, labels=df)
    print(error)
    print(var)
