import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    SST = np.sum(np.power((true - np.mean(true)), 2))
    SSReg = np.sum(np.power((pred - np.mean(true)), 2))
    return SSReg / SST


def cal_metric(pred, true, method):
    temp = 0
    for i in range(pred.shape[1]):
        temp += method(pred[:, i], true[:, i])
    temp /= pred.shape[1]
    return temp


def metric(pred, true):
    mae = cal_metric(pred, true, MAE)
    mse = cal_metric(pred, true, MSE)
    rmse = cal_metric(pred, true, RMSE)
    mape = cal_metric(pred, true, MAPE)
    mspe = cal_metric(pred, true, MSPE)
    r2 = cal_metric(pred, true, R2)
    return mae, mse, rmse, mape, mspe


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
