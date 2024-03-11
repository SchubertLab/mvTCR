"""
This trigonometric positional embedding was adapted from:
Title: Multigrate
Date: 17th March 2021
Availability: https://github.com/theislab/multigrate
"""
from torch import nn
import torch


class MLP(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 hiddens=None,
                 activation='relu',
                 output_activation='linear',
                 dropout=None,
                 batch_norm=True,
                 regularize_last_layer=False):
        super(MLP, self).__init__()

        # create network architecture
        layers = []
        if hiddens is None:  # no hidden layers
            layers.append(self._fc(n_inputs, n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   batch_norm=regularize_last_layer))
        else:
            layers.append(self._fc(n_inputs, hiddens[0], activation=activation, dropout=dropout, batch_norm=batch_norm))  # first layer
            for l in range(1, len(hiddens)):  # inner layers
                layers.append(self._fc(hiddens[l-1], hiddens[l], activation=activation, dropout=dropout, batch_norm=batch_norm))

            layers.append(self._fc(hiddens[-1], n_outputs, activation=output_activation,
                                   dropout=dropout if regularize_last_layer else None,
                                   batch_norm=regularize_last_layer))

        self.network = nn.Sequential(*layers)

    def _fc(self, n_inputs, n_outputs, activation='leakyrelu', dropout=None, batch_norm=True):
        layers = [nn.Linear(n_inputs, n_outputs, bias=not batch_norm)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        if activation != 'linear':
            layers.append(self._activation(activation))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def _activation(self, name='leakyrelu'):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace=True)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'softmax':
            return nn.Softmax()
        elif name == 'exponential':
            return Exponential()
        else:
            raise NotImplementedError(f'activation function {name} is not implemented.')

    def forward(self, x):
        return self.network(x)

    def through(self, x):
        outputs = []
        for layer in self.network:
            x = layer(x)
            outputs.append(x)
        return outputs


class Exponential(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x