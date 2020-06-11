import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as functional
from torch.autograd import Variable
import numpy as np
from .custom_layers import EncoderInput, DownConvolution, EncoderOutput
from .custom_layers import DecoderInput, UpConvolution, DecoderOutput
from .custom_layers import OneOneConvolution, AdvancedDecoderOutput
from torch.nn import init
import functools


class Connection(nn.Module):

    def __init__(self, num_layers=6, warp_channels=2):
        super(Connection, self).__init__()
        self.num_layers = num_layers

        one_one_list = list()
        for i in range(num_layers-3):
            one_one_list.append(OneOneConvolution(512, warp_channels if warp_channels else 2*512, True))

        self.one_one_list = nn.ModuleList(one_one_list)

    def forward(self, warp_list):
        container = list()
        for i in range(self.num_layers):
            out = self.one_one_list[i](warp_list[i+1])
            container.append(out)
        container.append(warp_list[-1])
        return container


class SkipConnectionEncode(nn.Module):
    def __init__(self, norm_layer="Batch", out_channel=512, num_layers=8):
        super(SkipConnectionEncode, self).__init__()

        if norm_layer == "Batch":
            self.use_bias = False
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.num_layers = num_layers
        down_list = list()
        down_list.append(EncoderInput(3, 64, self.use_bias))
        down_list.append(DownConvolution(64, 128, self.use_bias, self.norm_layer))
        down_list.append(DownConvolution(128, 256, self.use_bias, self.norm_layer))
        down_list.append(DownConvolution(256, 512, self.use_bias, self.norm_layer))
        for i in range(num_layers-5):
            down_list.append(DownConvolution(512, 512, self.use_bias, self.norm_layer))
        self.down_list = nn.ModuleList(down_list)
        self.out = EncoderOutput(512, out_channel, self.use_bias)

    def forward(self, x):
        skip_connection = list()
        out = x
        for i, module in enumerate(self.down_list):
            out = module(out)
            skip_connection.append(out)
        skip_connection.reverse()
        out = self.out(out)

        return out, skip_connection


class NoSkipConnectionDecode(nn.Module):

    def __init__(self, norm_layer="Batch", out_channel=2, num_layers=7, use_dropout=False, transpose=True):
        super(NoSkipConnectionDecode, self).__init__()

        if norm_layer == "Batch":
            self.use_bias = False
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        self.num_layers = num_layers
        up_list = list()
        up_list.append(DecoderInput(128, 512, self.use_bias, self.norm_layer, transpose))
        for i in range(num_layers-5):
            up_list.append(UpConvolution(512, 512, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(512, 256, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(256, 128, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(128, 64, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(DecoderOutput(64, out_channel, transpose))
        self.up_list = nn.ModuleList(up_list)

    def forward(self, x):
        warp_list = list()
        out = x
        warp_list.append(out)
        for i, module in enumerate(self.up_list):
            out = module(out)
            warp_list.append(out)

        return out, warp_list


class SkipConnectionDecode(nn.Module):

    def __init__(self, norm_layer="Batch", out_channel=3, num_layers=8,
                 use_dropout=False, transpose=True, use_advanced=False):
        super(SkipConnectionDecode, self).__init__()

        if norm_layer == "Instance":
            self.use_bias = True
            self.norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        elif norm_layer == "Batch":
            self.use_bias = False
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        elif norm_layer == 'Spectral':
            self.use_bias = True
            self.norm_layer = norm_layer

        self.use_advanced = use_advanced
        self.num_layers = num_layers
        up_list = list()
        up_list.append(DecoderInput(512, 512, self.use_bias, self.norm_layer, transpose))
        for i in range(num_layers-4):
            up_list.append(UpConvolution(1024, 512, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(1024, 256, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(512, 128, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(256, 64, self.use_bias, self.norm_layer, use_dropout, transpose))
        up_list.append(UpConvolution(128, 32, self.use_bias, self.norm_layer, use_dropout, transpose))

        up_list.append(DecoderOutput(32, out_channel, transpose))
        self.up_list = nn.ModuleList(up_list)
        if self.use_advanced:
            self.advanced = AdvancedDecoderOutput(128, out_channel, self.use_bias, self.norm_layer)

    def forward(self, x, skip_connection, warp_list=None, warp_list_flip=None):
        out_list = list()

        out = x.clone()
        out_list.append(out)

        for i, module in enumerate(self.up_list[:-1]):
            if i != 0:
                out = torch.cat((skip_connection[i-1], out), 1)
            out = module(out)
            out_list.append(out)

        out = self.up_list[-1](out)

        return out, out_list


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
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
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
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
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    return net


class Warper(nn.Module):
    def __init__(self, norm='Batch', warp_channels=2, num_layers=8,
                 use_dropout=False, transpose=False, warp_out=False, use_advanced=False):
        super(Warper, self).__init__()

        init_gain = 0.02
        init_type = 'xavier'

        """Create models
            Parameters:
                norm (str) -- the name of normalization layers used in the network: batch | instance | none
                warp_channels (bool) -- number of warp channels (if 0, warping each channels separately)
                num_layers (int) -- number of layers of skip connection encoder
                use_dropout (bool) -- if use dropout layers.
                transpose (bool) -- if use transpose convolution for up sampling
                init_type (str)    -- the name of our initialization method.
                init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        """


        driving_num_layers = num_layers - 1
        self.connection = init_weights(Connection(num_layers - 2, warp_channels), init_type, init_gain)

        self.encoder_d = init_weights(SkipConnectionEncode(norm, 512, num_layers),
                                       init_type, init_gain)
        self.decoder_d = init_weights(SkipConnectionDecode(norm, 2, driving_num_layers, use_dropout, transpose),
                                       init_type, init_gain)

    @staticmethod
    def flip_warp(warp_list):
        warp_list_flip = [0] * len(warp_list)
        for i, warper in enumerate(warp_list.copy()):

            warper_copy = warper.detach()
            # ones = torch.randn((1, warper_copy.size(-1), warper_copy.size(-1))).fill_(1)
            # flipper = torch.cat((-ones, ones), dim=0)
            #
            ones = np.ones((1, warper_copy.size(-1), warper_copy.size(-1)))
            flipper = np.concatenate((-ones, ones), 0)
            flipper = torch.Tensor(flipper).unsqueeze(0).repeat(warper_copy.size(0), 1, 1, 1)

            if torch.cuda.is_available():
                flipper = flipper.cuda()
            flipper = nn.Parameter(flipper, requires_grad=False)
            flipped_warper = flipper * warper_copy
            warp_list_flip[i] = flipped_warper
        return warp_list_flip

    def forward(self, pose, warper_flip=False):
        latent_d, skip_connection = self.encoder_d(pose)
        warp_output, warp_list = self.decoder_d(latent_d, skip_connection)

        return warp_output, warp_list

