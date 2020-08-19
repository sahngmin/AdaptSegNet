import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import numpy as np
import pdb

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

    def __init__(self, num_classes=1000, args=None):
        super(AlexNet_DM, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )

        self.last_feat = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.pooling = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True)
        )
        self.last_classifier = nn.Linear(128, num_classes)

        args.num_target = args.target_degrees.index(args.degree) + 1
        self.num_target = args.num_target
        for num in range(self.num_target):
            DM_name = 'DM' + str(num + 1)
            setattr(self, DM_name, DM(64, 64))  # without skip connection


    def DM_params(self, args):
        DM_name = 'DM' + str(args.num_target)

        b = []
        b.append(getattr(self, DM_name).parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        optim_parameters = [{'params': self.features.parameters(), 'lr': args.learning_rate},
                            {'params': self.last_feat.parameters(), 'lr': args.learning_rate},
                            {'params': self.classifier.parameters(), 'lr': args.learning_rate},
                            {'params': self.DM_params(args), 'lr': args.learning_rate},
                            {'params': self.last_classifier.parameters(), 'lr': args.learning_rate}]
        return optim_parameters


    def forward(self, x):
        x = self.features(x)
        out1 = x

        x2 = self.last_feat(x)
        DM_name = 'DM' + str(self.num_target)
        dm_out1, dm_out2 = getattr(self, DM_name)(out1)

        new_x = x2 + x2.view(x2.shape[0], 64, -1).std(dim=2, keepdim=True).unsqueeze(3) * \
                ((dm_out1 - dm_out1.view(x2.shape[0], 64, -1).mean(dim=2, keepdim=True).unsqueeze(3)) /
                 dm_out1.view(x2.shape[0], 64, -1).std(dim=2, keepdim=True).unsqueeze(3))
        new_x += x2.view(x2.shape[0], 64, -1).mean(dim=2, keepdim=True).std(dim=1, keepdim=True).unsqueeze(3) * \
                 (dm_out2 - dm_out2.view(x2.shape[0], -1).mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)) / \
                 dm_out2.view(x2.shape[0], -1).std(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)

        feat_ori = self.pooling(x2)
        feat_ori = torch.flatten(feat_ori, 1)
        feat_ori = self.classifier(feat_ori)
        output_ori = self.last_classifier(feat_ori)

        feat_new = self.pooling(new_x)
        feat_new = torch.flatten(feat_new, 1)
        feat_new = self.classifier(feat_new)
        output_new = self.last_classifier(feat_new)

        return feat_new, feat_ori, output_new, output_ori