import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Function


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


class DM(nn.Module):
    def __init__(self, in_channel=256, out_channel=256):
        super(DM, self).__init__()
        self.module_list = nn.ModuleList()
        self.module_list.append(conv1x1(in_channel, out_channel))
        self.module_list.append(nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                              nn.Conv2d(in_channel, out_channel, 1, bias=False),
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
        return out1, out2


class AlexNet_DM(nn.Module):

    def __init__(self, num_classes=10, num_target=1):
        super(AlexNet_DM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.last_feat = nn.Sequential(
            nn.Conv2d(64, 50, kernel_size=5, padding=2),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(100, num_classes)
        )

        self.num_target = num_target
        for num in range(self.num_target):
            DM_name = 'DM' + str(num + 1)
            setattr(self, DM_name, DM(64, 50))  # without skip connection


    def DM_params(self, args):
        DM_name = 'DM' + str(self.num_target)

        b = []
        b.append(getattr(self, DM_name).parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        if args.from_scratch:
            optim_parameters = [{'params': self.features.parameters(), 'lr': args.learning_rate},
                                {'params': self.last_feat.parameters(), 'lr': args.learning_rate},
                                {'params': self.classifier.parameters(), 'lr': args.learning_rate},
                                {'params': self.DM_params(args), 'lr': args.learning_rate},
                                {'params': self.last_classifier.parameters(), 'lr': args.learning_rate}]

        else:
            optim_parameters = [{'params': self.features.parameters(), 'lr': 0.01 * args.learning_rate},
                                {'params': self.last_feat.parameters(), 'lr': 0.01 * args.learning_rate},
                                {'params': self.classifier.parameters(), 'lr': 0.01 * args.learning_rate},
                                {'params': self.DM_params(args), 'lr': args.learning_rate},
                                {'params': self.last_classifier.parameters(), 'lr': 0.01 * args.learning_rate}]

        return optim_parameters


    def forward(self, x, num_target=None):
        x = self.features(x)
        out1 = x
        x2 = self.last_feat(x)
        if num_target is None:
            num_target = self.num_target
        DM_name = 'DM' + str(num_target)
        dm_out1, dm_out2 = getattr(self, DM_name)(out1)

        new_x = x2 + x2.view(x2.shape[0], 50, -1).std(dim=2, keepdim=True).unsqueeze(3) * \
                ((dm_out1 - dm_out1.view(x2.shape[0], 50, -1).mean(dim=2, keepdim=True).unsqueeze(3)) /
                 dm_out1.view(x2.shape[0], 50, -1).std(dim=2, keepdim=True).unsqueeze(3))
        new_x += x2.view(x2.shape[0], 50, -1).mean(dim=2, keepdim=True).std(dim=1, keepdim=True).unsqueeze(3) * \
                 (dm_out2 - dm_out2.view(x2.shape[0], -1).mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)) / \
                 dm_out2.view(x2.shape[0], -1).std(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)

        feat_ori = self.pooling(x2)
        output_ori = torch.flatten(feat_ori, 1)
        output_ori = self.classifier(output_ori)
        output_ori = self.last_classifier(output_ori)

        feat_new = self.pooling(new_x)
        output_new = torch.flatten(feat_new, 1)
        output_new = self.classifier(output_new)
        output_new = self.last_classifier(output_new)

        return feat_new, feat_ori, output_new, output_ori


class AlexNet_Source(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet_Source, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True)
        )

        self.last_feat = nn.Sequential(
            nn.Conv2d(64, 50, kernel_size=5, padding=2),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(50 * 4 * 4, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True)
        )
        self.last_classifier = nn.Sequential(
            nn.Linear(100, num_classes),
            nn.LogSoftmax()
        )


    def optim_parameters(self, args):
        optim_parameters = [{'params': self.features.parameters(), 'lr': args.learning_rate},
                            {'params': self.last_feat.parameters(), 'lr': args.learning_rate},
                            {'params': self.classifier.parameters(), 'lr': args.learning_rate},
                            {'params': self.last_classifier.parameters(), 'lr': args.learning_rate}]
        return optim_parameters


    def forward(self, x):
        x = self.features(x)
        out1 = x

        x2 = self.last_feat(x)

        feat_ori = self.pooling(x2)
        output_ori = torch.flatten(feat_ori, 1)
        output_ori = self.classifier(output_ori)
        output_ori = self.last_classifier(output_ori)

        return feat_ori, feat_ori, output_ori, output_ori


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANNModel(nn.Module):

    def __init__(self):
        super(DANNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5, padding=2))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.AdaptiveAvgPool2d((4, 4)))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        # input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output