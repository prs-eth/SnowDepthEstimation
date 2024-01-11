# Adapted from https://github.com/0zgur0/ms-convSTAR

import torch
import torch.nn as nn
# import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)


        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

        # print('convGRU cell is constructed with h_dim: ', hidden_size)


    def forward(self, input_, prev_state):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            if torch.cuda.is_available():
                prev_state = Variable(torch.zeros(state_size)).cuda()
            else:
                prev_state = Variable(torch.zeros(state_size))

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class ConvGRU(nn.Module):
    """
    Generates a multi-layer convolutional GRU.
    Preserves spatial dimensions across cells, only altering number of feature maps.

    Parameters:

        ``input_size``: integer. depth dimension of input tensors.

        ``hidden_sizes``: integer or list. depth dimensions of hidden state. If integer, the same hidden size is used for all cells.

        ``kernel_sizes``: integer or list. sizes of Conv2d gate kernels. If integer, the same kernel size is used for all cells.

        ``n_layers``: integer. number of chained ``ConvGRUCell``.
    """

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers=1):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        """

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.tail = nn.Sequential(nn.Conv2d(self.hidden_sizes[-1], 1, 1), nn.ReLU())


    def forward(self, x, hidden=None):
        """
        Forward pass.

        Parameters:
        
            ``x``: 4D input tensor. (batch, channels, height, width).
            
            ``hidden``: list of 4D hidden state representations. (batch, channels, height, width).

        Returns:

            ``upd_hidden``: 5D hidden states. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return tuple(upd_hidden)


    def forward_tail(self, hidden):
        """Produce outputs based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output snow depth maps
        """

        return self.tail(hidden[-1])




class ConvGRU_v(nn.Module):
    """
    Generates a multi-layer convolutional GRU.
    Preserves spatial dimensions across cells, only altering number of feature maps.

    Parameters:

        ``input_size``: integer. depth dimension of input tensors.

        ``hidden_sizes``: integer or list. depth dimensions of hidden state. If integer, the same hidden size is used for all cells.

        ``kernel_sizes``: integer or list. sizes of Conv2d gate kernels. If integer, the same kernel size is used for all cells.

        ``n_layers``: integer. number of chained ``ConvGRUCell``.
    """

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers=1):
        """
        Generates a multi-layer convolutional GRU for using with GaussianNLLLoss or LaplacianNLLLoss (i.e. log variance is estimated).
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        """

        super(ConvGRU_v, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.tail = nn.Sequential(nn.Conv2d(self.hidden_sizes[-1], 1, 1), nn.ReLU())
        # self.tail_var = nn.Sequential(nn.Conv2d(self.hidden_sizes[-1], 1, 1), nn.ReLU()) # negative variances would be clipped anyway in GaussianNLLLoss
        self.tail_var = nn.Conv2d(self.hidden_sizes[-1], 1, 1)


    def forward(self, x, hidden=None):
        """
        Forward pass.

        Parameters:
        
            ``x``: 4D input tensor. (batch, channels, height, width).
            
            ``hidden``: list of 4D hidden state representations. (batch, channels, height, width).

        Returns:

            ``upd_hidden``: 5D hidden states. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return tuple(upd_hidden)


    def forward_tail(self, hidden):
        """Produce outputs based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output snow depth maps
        """

        return self.tail(hidden[-1])


    def forward_tail_var(self, hidden):
        """Produce output variances based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output estimated heteroscedatic variances / uncertainties
        """

        return self.tail_var(hidden[-1])




class ConvGRU_Gamma(nn.Module):
    """
    Generates a multi-layer convolutional GRU.
    Preserves spatial dimensions across cells, only altering number of feature maps.

    Parameters:

        ``input_size``: integer. depth dimension of input tensors.

        ``hidden_sizes``: integer or list. depth dimensions of hidden state. If integer, the same hidden size is used for all cells.

        ``kernel_sizes``: integer or list. sizes of Conv2d gate kernels. If integer, the same kernel size is used for all cells.

        ``n_layers``: integer. number of chained ``ConvGRUCell``.
    """

    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers=1):
        """
        Generates a multi-layer convolutional GRU for using with Gamma NLL Loss.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        """

        super(ConvGRU_Gamma, self).__init__()

        self.input_size = input_size

        if type(hidden_sizes) != list:
            self.hidden_sizes = [hidden_sizes]*n_layers
        else:
            assert len(hidden_sizes) == n_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_sizes
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*n_layers
        else:
            assert len(kernel_sizes) == n_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i], self.kernel_sizes[i])
            name = 'ConvGRUCell_' + str(i).zfill(2)

            setattr(self, name, cell)
            cells.append(getattr(self, name))

        self.cells = cells

        self.tail_a = nn.Conv2d(self.hidden_sizes[-1], 1, 1)
        self.tail_b = nn.Conv2d(self.hidden_sizes[-1], 1, 1)


    def forward(self, x, hidden=None):
        """
        Forward pass.

        Parameters:
        
            ``x``: 4D input tensor. (batch, channels, height, width).
            
            ``hidden``: list of 4D hidden state representations. (batch, channels, height, width).

        Returns:

            ``upd_hidden``: 5D hidden states. (layer, batch, channels, height, width).
        """
        if not hidden:
            hidden = [None]*self.n_layers

        input_ = x

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(input_, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            input_ = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return tuple(upd_hidden)


    def forward_tail_a(self, hidden):
        """Produce outputs based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output snow depth maps
        """

        return self.tail_a(hidden[-1])


    def forward_tail_b(self, hidden):
        """Produce output variances based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output estimated heteroscedatic variances / uncertainties
        """

        return self.tail_b(hidden[-1])


    def forward_tail_mean_estimate(self, hidden):
        """Produce output variances based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output estimated heteroscedatic variances / uncertainties
        """
        
        a = self.tail_a(hidden[-1])
        b = self.tail_b(hidden[-1])

        # return torch.exp(b) + torch.exp(a + b) # rep 1
        return torch.exp(a + b) # rep 2


    def forward_tail_var_estimate(self, hidden):
        """Produce output variances based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output estimated heteroscedatic variances / uncertainties
        """
        
        a = self.tail_a(hidden[-1])
        b = self.tail_b(hidden[-1])

        # return torch.exp(2*b) + torch.exp(a + 2*b) # rep 1
        return torch.exp(a + 2*b) # rep 2


    def forward_tail_mode_estimate(self, hidden):
        """Produce output mode (maximum likelihood estimator) based on the last layer's hidden state.

        Parameters:

            ``hidden``: list of hidden states

        Returns:

            Output estimated heteroscedatic variances / uncertainties
        """
        
        a = self.tail_a(hidden[-1])
        b = self.tail_b(hidden[-1])

        # return torch.exp(a + b) # rep 1

        mode = (torch.exp(a) - 1) * torch.exp(b) # rep 2
        mode[a <= 0] = 0 # rep 2
        return mode # rep 2

