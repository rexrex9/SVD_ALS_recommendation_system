
import numpy as np
import torch

def MSE(y_true, y_pred):
    return np.average((np.array(y_true) - np.array(y_pred)) ** 2)

def RMSE(y_true, y_pred):
    return MSE(y_true, y_pred) ** 0.5

def doEva( net, d ):
    d = torch.LongTensor(d)
    u, i, r = d[:, 0], d[:, 1], d[:, 2]
    with torch.no_grad():
        y_pred = net(u,i)
    y_true = r.detach().numpy()
    return RMSE(y_true, y_pred)