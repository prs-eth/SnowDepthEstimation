import torch
import torch.nn as nn
from torch.nn import init
# import functools
from torch.optim import lr_scheduler
# from math import ceil

from models.convgru import ConvGRU, ConvGRU_v, ConvGRU_Gamma
# from models.convlstm import ConvLSTM
# from models.convstar import ConvSTAR

# from models.stochastic_input import SIConvGRU

# import models.STCNN as STCNN

# from models.RUN import RUN

###############################################################################
# Helper Functions
###############################################################################


class Identity(nn.Module):
    def forward(self, x):
        return x


# def get_norm_layer(norm_type='instance'):
#     """Return a normalization layer

#     Parameters:
#         norm_type (str) -- the name of the normalization layer: batch | instance | none

#     For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
#     For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
#     """
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         norm_layer = lambda x: Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.3162, verbose=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1, verbose=True)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, min_lr=1e-4, patience=25, verbose=True)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='none', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'none':
                pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 and init_type != 'none':  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='none', init_gain=0.02, gpu_ids=[], skip_init=False):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    if not skip_init:
        init_weights(net, init_type, init_gain=init_gain)
    return net


def get_model(
        rnn_type=None,
        input_size=None,
        hidden_sizes=None,
        kernel_sizes=None,
        n_layers=None,
        patch_size=None,
        init_type='none',
        init_gain=0.02,
        gpu_ids=[0],
        sequence_length=10,
        partial=False,
        efficientnet_level=0,
        run_recurrent_unit='GRU',
    ):
    """Create an RNN model
    """
    net = None

    if rnn_type == 'ConvGRU':
        net = ConvGRU(input_size, hidden_sizes, kernel_sizes, n_layers=n_layers)
    elif rnn_type == 'ConvGRU_v':
        net = ConvGRU_v(input_size, hidden_sizes, kernel_sizes, n_layers=n_layers)
    # elif rnn_type == 'ConvGRU_Gamma':
    #     net = ConvGRU_Gamma(input_size, hidden_sizes, kernel_sizes, n_layers=n_layers)
    # elif rnn_type == 'ConvLSTM':
    #     net = ConvLSTM((patch_size, patch_size), input_size, hidden_sizes, (kernel_sizes, kernel_sizes), n_layers)
    # elif rnn_type == 'ConvSTAR':
    #     net = ConvSTAR(input_size, hidden_sizes, kernel_sizes, n_layers)
    # elif rnn_type == 'TempCNN':
    #     net = STCNN.TempCNN(in_ch=input_size, out_ch=1, sequence_length=sequence_length, partial=partial)
    # elif rnn_type == 'STCNN':
    #     net = STCNN.STCNN(in_ch=input_size, out_ch=1, sequence_length=sequence_length, partial=partial)
    # elif rnn_type == 'RUN':
    #     net = RUN(
    #     encoder_name=f'efficientnet-b{str(efficientnet_level)}',
    #     in_channels = input_size,
    #     classes = 1,
    #     recurrent_unit = run_recurrent_unit,
    # )
    # elif rnn_type == 'SIConvGRU':
    #     net = SIConvGRU(input_size, hidden_sizes, kernel_sizes, n_layers=n_layers)
    else:
        raise NotImplementedError('Network model name [%s] is not recognized' % rnn_type)
    return init_net(net, init_type, init_gain, gpu_ids, skip_init=rnn_type=='RUN')

