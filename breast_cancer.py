from sklearn.datasets import load_breast_cancer as lbc
import torch
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam
from models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import training_utils as utils

device = torch.device("cpu")


X, y = lbc(return_X_y=True)
X = torch.tensor(X)
y = torch.tensor(y)
data = utils.CustomDataObject(X, y)
data = torch.utils.data.DataLoader(data, batch_size=1)
print(X.shape)
print(y.shape)

def categorical(y):
    N = len(y)
    M = np.max(y) + 1
    Y = np.zeros((N, M))
    for i in range(N):
        Y[i, y[i]] = 1
    return Y


def run_model(model, optimizer, train_loader, device, num_epochs=50):
    # pbar = tqdm(range(num_epochs))
    for i in range(num_epochs):
        for bi, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            out = model(x, do_kl=False)
            # print(out)
            out = F.softmax(out, dim=1)
            # print(out)


            loss = -F.nll_loss(out, y) * 100
            loss.backward()
            optimizer.step()

        print(f"epoch: {i}, loss: {loss}")
        # exit()
        #

model = BayesianFC(X.shape[1], 2, [20, 20])
model.freeze()
optim = Adam(model.parameters(), lr=0.001)

run_model(model, optim, data, device)

pred = model(X, do_kl=False)
out = torch.argmax(F.softmax(pred, dim=1), dim=1)

print(y)
print(out)

print(torch.sum(y == out) / len(y))

