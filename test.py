import torch
import torch.nn.functional as F
from torch.optim import Adam
# from models.Bayes3FC import Bayesian3FC
# from models.F3FC import F3FC
from models import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import training_utils as utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# class Data:
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

#     def __len__(self):
#         return self.x.size(0)


# data_file = "DATA/carbon_nanotubes.csv"
# data = pd.read_csv(data_file, sep=";", decimal=",")
# target_label = "Calculated atomic coordinates w"
# inputs_labels = [col for col in data.columns if target_label not in col]
# target_labels = [col for col in data.columns if target_label in col]
# X = torch.tensor(data[inputs_labels].values).type(torch.float)
# Y = torch.tensor(data[target_labels].values).type(torch.float)

# inds = np.arange(X.size(0))
# torch.manual_seed(12345)
# np.random.seed(12345)
# np.random.shuffle(inds)
# test_size = 0.2
# batch_size = 20
# test_len = int(test_size * X.size(0))
# train_inds, test_inds = inds[test_len:], inds[:test_len]

# X_train, X_test = X[train_inds], X[test_inds]
# Y_train, Y_test = Y[train_inds], Y[test_inds]

# train_loader = torch.utils.data.DataLoader(Data(X_train, Y_train), batch_size=batch_size)

# lr_start = 0.0001
# # model = F3FC(7, 30, 20, 1)
# torch.manual_seed(123)
train_loader, test_data = utils.get_nanotube_data()
X_test, Y_test = test_data

print(Y_test.std())
# model = Bayesian3FC(7, 30, 20, 1, prior={"dist": "gaussian", "params": {"mean": 0, "std": 0.01}})
# test = BayesianFC(7, 1, hidden_nodes=[30, 20], prior={"dist": "gaussian", "params": {"mean": 0, "std": 0.01}})
# x = torch.randn((10, 7)).to(torch.double)

# y_old, kl1 = model(x)
# y_new, kl2 = test(x)
# print(y_old.mean(), y_old.std())
# print(y_new.mean(), y_new.std())
# print(kl1, kl2)
# optimizer = Adam(model.parameters(), lr=lr_start)


# num_epochs = 30
# pbar = tqdm(range(num_epochs))

# for i in pbar:
#     for batch_index, (x, y) in enumerate(train_loader):
#         x, y = x.to(device), y.to(device)
#         out = model(x)

#         loss = F.mse_loss(out, y)
#         loss.backward()
#         optimizer.step()

#     pbar.set_description(f"{loss}")

# plt.plot(model.fc1.means)
# plt.show()
# plt.plot(model.fc2.means)
# plt.show()
# plt.plot(model.fc3.means)
# plt.show()

# Y_sample = Y_test.detach().numpy()[:20]
# Y_pred_sample = model(X_test).detach().numpy()[:20]
# plt.plot(Y_sample, "bo-", ms=8, label="target")
# plt.plot(Y_pred_sample, "ro--", ms=4, label="prediction")
# plt.legend()
# plt.show()