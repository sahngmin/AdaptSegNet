import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import torch.nn.functional as functional
from torch.autograd import Variable
from model.warper import Warper
from collections import OrderedDict
import operator
from itertools import islice

affine_par = True

def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

class Multioutput_Sequential(nn.Module):
    def __init__(self, *args):
        super(Multioutput_Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(Multioutput_Sequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input[0])
        return input

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        nonlinear_out = self.relu(out)

        return [nonlinear_out, out]


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

# class DM(nn.Module):
#     def __init__(self, inplanes, dilation_series, padding_series, num_classes):
#         super(DM, self).__init__()
#         self.conv2d_list = nn.ModuleList()
#         for dilation, padding in zip(dilation_series, padding_series):
#             self.conv2d_list.append(
#                 nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
#
#         for m in self.conv2d_list:
#             m.weight.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         out = self.conv2d_list[0](x)
#         for i in range(len(self.conv2d_list) - 1):
#             out += self.conv2d_list[i + 1](x)
#         return out

class DM(nn.Module):
    def __init__(self, inplanes, num_classes):
        super(DM, self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(conv1x1(inplanes, num_classes))
        self.module_list.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(inplanes, num_classes, 1, bias=False),
                                              nn.ReLU()))

        for i, m in enumerate(self.module_list):
            if i == 0:
                m.weight.data.normal_(0, 0.01)
            else:
                for n in m:
                    if isinstance(n, nn.Conv2d):
                        n.weight.data.normal_(0, 0.01)
                    elif isinstance(n, nn.BatchNorm2d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()

    def forward(self, x):
        out1 = self.module_list[0](x)
        out2 = self.module_list[1](x)
        # out2 = functional.interpolate(self.module_list[1](x), size=out1.size()[2:], mode='bilinear', align_corners=True)
        return out1, out2

class ResNet_DM(nn.Module):
    def __init__(self, block, layers, num_classes, args=None, len_dataset=None):
        super(ResNet_DM, self).__init__()
        self.memory = args.memory
        self.warper = args.warper

        self.num_dataset = args.num_dataset
        self.num_classes = num_classes
        self.batch_size = args.batch_size

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        if args.warper:
            self.WarpModel = Warper(args=args)

        if args.memory:
            for num_dataset in range(len_dataset):
                DM_name = 'DM' + str(num_dataset + 1)
                # setattr(self, DM_name, self._make_pred_layer(DM, 3072, [6, 12], [6, 12], num_classes))
                setattr(self, DM_name, DM(2048, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return Multioutput_Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)


    def forward(self, image, input_size, map=None):
        output_both_warped, output_ori_warped, output_both, output_ori = None, None, None, None

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1([x])
        x = self.layer2(x)
        x = self.layer3(x)

        x1 = self.layer4(x)

        x2 = self.layer6(x1[0])
        output_ori = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(x2)  # ResNet + ASPP

        if self.memory:
            DM_name = 'DM' + str(self.num_dataset)

            # x3 = torch.cat((x[1], x1[1]), 1)  # concatenate linear(1) outputs
            x3 = x1[0]  # without skip connection nonlinear(0)/linear(1) output

            x3_1, x3_2 = getattr(self, DM_name)(x3)
            new_x = x2 + x2.view(self.batch_size, self.num_classes, -1).std(dim=2, keepdim=True).unsqueeze(3) * \
                    ((x3_1 - x3_1.view(self.batch_size, self.num_classes, -1).mean(dim=2, keepdim=True).unsqueeze(3)) /
                     x3_1.view(self.batch_size, self.num_classes, -1).std(dim=2, keepdim=True).unsqueeze(3))
            # new_x += x2.view(self.batch_size, self.num_classes, -1).mean(dim=2, keepdim=True).std(dim=1, keepdim=True).unsqueeze(3) * \
            #          (x3_2 - x3_2.view(self.batch_size, -1).mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)) / \
            #          x3_2.view(self.batch_size, -1).std(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
            new_x += functional.interpolate(x3_2, size=new_x.size()[2:], mode='bilinear', align_corners=True)
            output_both = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)(new_x)  # ResNet + (ASPP+DM)

        if self.warper:
            if map is not None:
                warper, warp_list = self.WarpModel(image, map)
            else:
                # x2_softmax = nn.Softmax()(x2_up)
                # x2_filtered = self.threshold(x2_softmax)
                # warper, warp_list = self.WarpModel(image, x2_filtered)

                if not self.memory:
                    warper, warp_list = self.WarpModel(output_ori, None)
                    output_ori_warped = self.warp(output_ori, warper)
                else:
                    warper, warp_list = self.WarpModel(output_both, None)
                    output_both_warped = self.warp(output_both, warper)

        return output_both_warped, output_ori_warped, output_both, output_ori


    def ResNet_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def ASPP_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def DM_params(self, args):
        DM_name = 'DM' + str(args.num_dataset)

        b = []
        b.append(getattr(self, DM_name).parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def parameters_seg(self, args):
        if args.num_dataset == 1 or args.source_only:
            optim_parameters = [{'params': self.ResNet_params(), 'lr': args.learning_rate},
                                {'params': self.ASPP_params(), 'lr': 10 * args.learning_rate}]

        else:
            optim_parameters = [{'params': self.ResNet_params(), 'lr': args.learning_rate},
                                {'params': self.ASPP_params(), 'lr': args.learning_rate}]

        if self.memory:
            optim_parameters += [{'params': self.DM_params(args), 'lr': 10 * args.learning_rate}]

        return optim_parameters

    def parameters_warp(self, args):
        optim_parameters = [{'params': self.WarpModel.parameters(), 'lr': 10 * args.learning_rate}]
        return optim_parameters

    @staticmethod
    def warp(input, warper):

        xs1 = np.linspace(-1, 1, input.size(2))
        xs2 = np.linspace(-1, 1, input.size(3))
        xs = np.meshgrid(xs2, xs1)
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(input.size(0), 1, 1, 1)
        if torch.cuda.is_available():
            xs = xs.cuda()

        sampled = torch.zeros(input.size()).cuda()
        for i in range(warper.size(1) // 2):
            sampler = 0.2 * nn.Tanh()(warper[:, i * 2:(i + 1) * 2, :, :]).permute(0, 2, 3, 1) + Variable(xs, requires_grad=False)
            sampler = sampler.clamp(min=-1, max=1)
        sampled = functional.grid_sample(input, sampler)

        return sampled


def Deeplab_DM(args=None):
    model = ResNet_DM(Bottleneck, [3, 4, 23, 3], num_classes=args.num_classes, args=args, len_dataset=args.num_dataset)
    return model




