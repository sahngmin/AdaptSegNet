import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import copy
from options import TrainOptions, dataset_dict
from model.deeplab_DM import Deeplab_DM
from model.discriminator import FCDiscriminator
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.synthia_dataset import synthiaDataset
from utils.tsne_plot import TSNE_plot
from utils.model_save import save_model
from torch.nn import DataParallel



PRE_TRAINED_SEG = './snapshots/OLD/Scratch_warper/single_level/GTA5_75000.pth'
# PRE_TRAINED_DISC = './snapshots/GTA2Cityscape/GTA5toCityScapes_single_level_best_model_D.pth'
PRE_TRAINED_DISC = None

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

args = TrainOptions().parse()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        if args.num_dataset == 1:
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
        else:
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr


def distillation_loss(pred_origin, old_outputs):
    pred_origin_logsoftmax = (pred_origin / 2).log_softmax(dim=1)
    old_outputs = (old_outputs / 2).softmax(dim=1)
    loss_distillation = (-(old_outputs * pred_origin_logsoftmax)).sum(dim=1)
    loss_distillation = loss_distillation.sum() / loss_distillation.flatten().shape[0]
    return loss_distillation


def main():
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    model = Deeplab_DM(args=args)
    model_D = None

    if args.source_only:  # training model from pre-trained ResNet on source domain
        if args.restore_from_resnet[:4] == 'http':
            saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
        else:
            saved_state_dict = torch.load(args.restore_from_resnet)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:
        if args.from_scratch:  # training model from pre-trained ResNet on source & target domain
            if args.restore_from_resnet[:4] == 'http':
                saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
            else:
                saved_state_dict = torch.load(args.restore_from_resnet)

            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:  # training model from pre-trained DeepLab on source & target domain
            saved_state_dict = torch.load(PRE_TRAINED_SEG, map_location=device)
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                if i in new_params.keys():
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)

        model_D = FCDiscriminator(num_classes=args.num_classes).to(device)

        if PRE_TRAINED_DISC is not None:
            saved_state_dict_D = torch.load(PRE_TRAINED_DISC, map_location=device)
            new_params = model_D.state_dict().copy()
            for i in saved_state_dict_D:
                if i in new_params.keys():
                    new_params[i] = saved_state_dict_D[i]
            model_D.load_state_dict(new_params)

        model_D.train()
        if args.multi_gpu:
            model_D = DataParallel(model_D)
            optimizer_D = optim.Adam(model_D.module.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        else:
            model_D.to(device)
            optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

        if args.target == 'CityScapes':
            targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.batch_size,
                                                             crop_size=input_size_target,
                                                             scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                             set=args.set),
                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        elif args.target == 'Synthia':  # ------------------------SYNTHIA dataloader 필요!!!
            targetloader = data.DataLoader(
                synthiaDataset("SYNTHIA-SEQS-01-WINTERNIGHT", 'synthia_01winternight_list/train.txt',
                               max_iters=args.num_steps * args.batch_size,
                               crop_size=input_size_target,
                               scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                               set=args.set),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                pin_memory=True)
        else:
            raise NotImplementedError('Unavailable target domain')

        targetloader_iter = enumerate(targetloader)

        optimizer_D.zero_grad()

        if args.gan == 'Vanilla':
            bce_loss = torch.nn.BCEWithLogitsLoss()
        elif args.gan == 'LS':
            bce_loss = torch.nn.MSELoss()

        # labels for adversarial training
        source_label = 0
        target_label = 1

    model.train()

    if args.multi_gpu:
        model = DataParallel(model)
        # implement model.optim_parameters(args) to handle different models' lr setting
        optimizer = optim.SGD(model.parameters_seg_multi(args),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        model.to(device)
        # implement model.optim_parameters(args) to handle different models' lr setting
        optimizer = optim.SGD(model.parameters_seg(args),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if not args.source_only and args.memory and not args.num_dataset == 1:
        ref_model = copy.deepcopy(model)  # reference model for distillation loss
        for params in ref_model.parameters():
            params.requires_grad = False

    cudnn.benchmark = True
    optimizer.zero_grad()

    if args.warper:
        optimizer_warp = optim.Adam(model.parameters_warp(args), lr=args.learning_rate)
        optimizer_warp.zero_grad()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

        # Dataloader
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    # start training
    for i_iter in range(args.num_steps):

        loss_seg_value_before_warped = 0
        loss_seg_value_after_warped = 0
        loss_distillation_value = 0
        loss_adv_target_value = 0
        loss_D_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        if not args.source_only:
            optimizer_D.zero_grad()
            adjust_learning_rate_D(optimizer_D, i_iter)

        if args.warper:
            optimizer_warp.zero_grad()
            adjust_learning_rate(optimizer_warp, i_iter, args)

        # train G

        if not args.source_only:
            for param in model_D.parameters():
                param.requires_grad = False

        _, batch = trainloader_iter.__next__()

        images, labels, _, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        if args.warper and args.memory:
            pred_warped, _, pred, pred_ori = model(images, input_size)
        elif args.warper:
            _, pred_warped, _, pred = model(images, input_size)
        elif args.memory:
            _, _, pred, pred_ori = model(images, input_size)
        else:
            _, _, _, pred = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value_before_warped += loss_seg.item()

        if not args.source_only and args.memory and not args.num_dataset == 1:
            _, _, _, old_outputs = ref_model(images, input_size)
            loss_distillation = distillation_loss(pred_ori, old_outputs)
            loss += args.lambda_distillation * loss_distillation
            loss_distillation_value += loss_distillation.item()

        loss.backward()

        if not args.source_only:
            _, batch = targetloader_iter.__next__()
            images_target, _, _ = batch
            images_target = images_target.to(device)

            if args.warper and args.memory:
                pred_target_warped, _, pred_target, _ = model(images_target, input_size)
            elif args.warper:
                _, pred_target_warped, _, pred_target = model(images_target, input_size)
            elif args.memory:
                _, _, pred_target, _ = model(images_target, input_size)
            else:
                _, _, _, pred_target = model(images_target, input_size)


            D_out = model_D(F.softmax(pred_target, dim=1))

            loss_adv_target = bce_loss(D_out,
                                       torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv_target[args.num_dataset - 1] * loss_adv_target
            loss_adv_target_value += loss_adv_target.item()
            loss.backward()

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred.detach()

            D_out = model_D(F.softmax(pred, dim=1))

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
            loss_D = loss_D / 2

            loss_D.backward()

            loss_D_value += loss_D.item()

            # train with target
            pred_target = pred_target.detach()

            D_out = model_D(F.softmax(pred_target, dim=1))

            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))

            loss_D = loss_D / 2

            loss_D.backward()
            loss_D_value += loss_D.item()

        optimizer.step()
        if not args.source_only:
            optimizer_D.step()

        if args.warper:  # retain_graph=True -> CUDA out of memory
            if args.memory:  # feed-forwarding the input again
                pred_warped, _, pred, _ = model(images, input_size)
            else:
                _, pred_warped, _, pred = model(images, input_size)

            loss_seg = seg_loss(pred_warped, labels)
            loss = loss_seg
            loss_seg_value_after_warped += loss_seg.item()
            loss.backward()
            optimizer_warp.step()

        if args.tensorboard:
            scalar_info = {
                'loss_seg_before_warped': loss_seg_value_before_warped,
                'loss_seg_after_warped': loss_seg_value_after_warped,
                'loss_distillation': loss_distillation_value,
                'loss_adv_target': loss_adv_target_value,
                'loss_D': loss_D_value
                }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg_before_warped = {2:.3f} loss_seg_after_warped = {3:.3f} loss_distill = {4:.3f} loss_adv = {5:.3f} loss_D = {6:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value_before_warped, loss_seg_value_after_warped, loss_distillation_value,
                loss_adv_target_value, loss_D_value))

        state = save_model(i_iter, args, model, model_D)
        if state:
            break

    if args.tensorboard:
        writer.close()

if __name__ == '__main__':
    main()
