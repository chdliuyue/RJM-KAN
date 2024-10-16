from SM_data import X_TRAIN, Q_TRAIN, y_TRAIN, X_TEST, Q_TEST, y_TEST
import torch
import torch.nn as nn
import numpy as np


def datapre(d: int):
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embeddings = nn.Embedding(81, 1).to(device)
    nn.init.kaiming_normal_(embeddings.weight, a=0, mode='fan_in', nonlinearity='relu')
    embeddings.weight.requires_grad = False

    train_input_temp = X_TRAIN[:, 2:, :, :]
    train_input_temp = train_input_temp.reshape(-1, 9)
    train_X = train_input_temp[:, :8]
    train_Q_temp = embeddings(torch.from_numpy(Q_TRAIN).to(device))
    train_Q = train_Q_temp.squeeze(-1).cpu().detach().numpy()

    test_input_temp = X_TEST[:, 2:, :, :]
    test_input_temp = test_input_temp.reshape(-1, 9)
    test_X = test_input_temp[:, :8]
    test_Q_temp = embeddings(torch.from_numpy(Q_TEST).to(device))
    test_Q = test_Q_temp.squeeze(-1).cpu().detach().numpy()
    if d == 8:
        train_input = train_X
        test_input = test_X
        print("use Continuous Dataset and feature dimension is ", train_input.shape)
    elif d == 12:
        train_input = train_Q
        test_input = test_Q
        print("use Discrete Dataset and feature dimension is ", train_input.shape)
    else:
        train_input = np.concatenate((train_X, train_Q), axis=1)
        test_input = np.concatenate((test_X, test_Q), axis=1)
        print("use Continuous and Discrete Dataset and feature dimension is ", train_input.shape)

    train_label = np.argmax(y_TRAIN, axis=1)
    test_label = np.argmax(y_TEST, axis=1)

    dataset = {}
    dtype = torch.get_default_dtype()
    dataset['train_input'] = torch.from_numpy(train_input).type(dtype).to(device)
    dataset['test_input'] = torch.from_numpy(test_input).type(dtype).to(device)
    dataset['train_label'] = torch.from_numpy(train_label).type(torch.long).to(device)
    dataset['test_label'] = torch.from_numpy(test_label).type(torch.long).to(device)

    return dataset




