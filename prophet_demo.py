import pandas as pd
from fbprophet import Prophet

from utils.metrics import *
from tqdm import tqdm

def prophet(ahead, start_exp, n_samples, labels):
    var = []
    for idx in range(ahead):
        var.append([])
    loss = []
    temp_loss = []
    mae_loss = []
    rmse_loss = []
    count = 0
    pred = []
    real = []
    Y = []
    REAL = []
    for test_sample in tqdm(range(start_exp, n_samples - ahead), 'training:'):  #
        print(test_sample)
        count += 1
        for j in range(labels.shape[1] - 2):
            ds = labels.iloc[count:test_sample, [1, j + 2]]
            ds['Time'] = pd.to_datetime(ds['Time'])
            ds.columns = ["ds", "y"]
            # with suppress_stdout_stderr():
            m = Prophet(interval_width=0.95)
            m.fit(ds)
            future = m.predict(m.make_future_dataframe(periods=ahead, freq='h'))
            yhat = float(future["yhat_lower"].tail(ahead).values)
            y_me = float(labels.iloc[test_sample:test_sample + ahead, j + 2].values)
            Y.append(yhat)
            REAL.append(y_me)

        rmse = RMSE(np.array(Y), np.array(REAL))
        mae = MAE(np.array(Y), np.array(REAL))
        rmse_loss.append(rmse)
        mae_loss.append(mae)
    print((sum(rmse_loss) / len(rmse_loss)))
    print((sum(mae_loss) / len(mae_loss)))


if __name__ == "__main__":
    df = pd.read_csv('new.csv')
    prophet(ahead=1, start_exp=72, n_samples=100, labels=df)
