import torch
from torch.nn import functional as F
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


class CustomDataObject:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.size(0)


def get_nanotube_data(test_size=0.2, batch_size=20, target_label=None, seed=12345):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # read data from file
    data_file = "DATA/carbon_nanotubes.csv"
    data = pd.read_csv(data_file, sep=";", decimal=",")
    data = (data-data.mean(axis=0))/data.std(axis=0)
    
    if target_label is None:
        target_label = "Calculated atomic coordinates w"
    # columns to use as input and target
    inputs_labels = [col for col in data.columns if target_label not in col]
    target_labels = [col for col in data.columns if target_label in col]
    X = torch.tensor(data[inputs_labels].values).type(torch.double)  # enforce datatype as doubles
    Y = torch.tensor(data[target_labels].values).type(torch.double)

    # random shuffle of data
    inds = np.arange(X.size(0))
    np.random.shuffle(inds)
    test_len = int(test_size * X.size(0))
    train_inds, test_inds = inds[test_len:], inds[:test_len]

    # splint in train and test
    X_train, X_test = X[train_inds], X[test_inds]
    Y_train, Y_test = Y[train_inds], Y[test_inds]

    # Package data for easy training and testing
    train_loader = torch.utils.data.DataLoader(CustomDataObject(X_train, Y_train), batch_size=batch_size)
    test_data = (X_test, Y_test)
    return train_loader, test_data


def train_model(model, optimizer, train_loader, device, num_epochs=30):
    # num_epochs = num_epochs
    # if loss == 'mse':
    #     Loss_FN = torch.nn.MSELoss()
    # else:
    #     print('Not Recgonized Loss Type')
    #     return
    pbar = tqdm(range(num_epochs))
    for i in pbar:
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            kl = model.total_kl_div

            loss = -torch.distributions.Normal(out, .1).log_prob(y).sum() + kl
            loss.backward()

            model.total_kl_div = 0
            optimizer.step()

        pbar.set_description(f"{loss}")



