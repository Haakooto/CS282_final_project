import torch
import pandas as pd
import numpy as np


class CustomDataObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.size(0)


def get_nanotube_data(test_size=0.2, batch_size=20, target_label=None, scale=True, seed=12345):
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # read data from file
    data_file = "DATA/carbon_nanotubes.csv"
    data = pd.read_csv(data_file, sep=";", decimal=",")

    if target_label is None:
        target_label = "Calculated atomic coordinates w"
    # columns to use as input and target
    inputs_labels = [col for col in data.columns if target_label not in col]
    target_labels = [col for col in data.columns if target_label in col]
    X = torch.tensor(data[inputs_labels].values).type(
        torch.double)  # enforce datatype as doubles
    Y = torch.tensor(data[target_labels].values).type(torch.double)

    # random shuffle of data
    inds = np.arange(X.size(0))
    np.random.shuffle(inds)
    test_len = int(test_size * X.size(0))
    train_inds, test_inds = inds[test_len:], inds[:test_len]

    # splint in train and test
    X_train, X_test = X[train_inds], X[test_inds]
    Y_train, Y_test = Y[train_inds], Y[test_inds]

    if scale:
        # standard scaling
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        # Y_train = (Y_train - Y_train.mean(axis=0)) / Y_train.std(axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
        # Y_test = (Y_test - Y_test.mean(axis=0)) / Y_test.std(axis=0)

    # Package data for easy training and testing
    train_loader = torch.utils.data.DataLoader(
        CustomDataObject(X_train, Y_train), batch_size=batch_size)
    test_data = (X_test, Y_test)
    return train_loader, test_data
