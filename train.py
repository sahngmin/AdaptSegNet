import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import copy
from options import TrainOptions, dataset_list
from model.deeplab_DM import Deeplab_DM
from model.discriminator import FCDiscriminator, Hinge, SpectralDiscriminator
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.synthia_dataset import SYNTHIADataSet
from utils.tsne_plot import TSNE_plot
from utils.custom_function import save_model, load_existing_state_dict
from torch.nn import DataParallel


torch.manual_seed(0) # cpu 연산 무작위 고정
torch.cuda.manual_seed(0) # gpu 연산 무작위 고정
torch.cuda.manual_seed_all(0) # 멀티 gpu 연산 무작위 고정
torch.backends.cudnn.enabled = False # cudnn library를 사용하지 않게 만듬
np.random.seed(0) # numpy 관련 연산 무작위 고정
random.seed(0)


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
        if args.num_dataset == 2:
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
    # device = torch.device('cpu')

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    # cudnn.enabled = True

    # Create network
    model = Deeplab_DM(args=args, device=device)
    optimizer = optim.SGD(model.parameters_seg(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    if (args.warper or args.spadeWarper):
        optimizer_warp = optim.Adam(model.parameters_warp(args), lr=args.learning_rate)
    else:
        optimizer_warp = None
    # model_D = FCDiscriminator(num_classes=args.num_classes)
    model_D = SpectralDiscriminator(num_classes=args.num_classes)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))


    checkpoint = None
    if checkpoint is not None:
        checkpoint_file = torch.load(checkpoint)
        model = load_existing_state_dict(model, checkpoint_file['state_dict_seg'])
        model_D = load_existing_state_dict(model_D, checkpoint_file['discriminator'])
        optimizer = load_existing_state_dict(optimizer, checkpoint_file['optimizer_seg'])
        optimizer_D = load_existing_state_dict(optimizer_D, checkpoint_file['optimizer_disc'])
        if checkpoint_file['optimizer_warp'] is not None:
            optimizer_warp = load_existing_state_dict(optimizer_warp, checkpoint_file['optimizer_warp'])
        del checkpoint_file


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

    if PRE_TRAINED_DISC is not None:
        saved_state_dict_D = torch.load(PRE_TRAINED_DISC, map_location=device)
        new_params = model_D.state_dict().copy()
        for i in saved_state_dict_D:
            if i in new_params.keys():
                new_params[i] = saved_state_dict_D[i]
        model_D.load_state_dict(new_params)

    if args.multi_gpu:
        model = DataParallel(model)
        model_D = DataParallel(model_D)

    else:
        model.to(device)
        model_D.to(device)


        # Dataloader
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size, crop_size=input_size, set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    if args.target == 'CityScapes':
        targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target, max_iters=args.num_steps * args.batch_size,
                                                         crop_size=input_size_target, set=args.set),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)
    elif args.target == 'Synthia':  # ------------------------SYNTHIA dataloader 필요!!!
        targetloader = data.DataLoader(SYNTHIADataSet(args.data_dir_target, args.data_list_target,
                                    max_iters=args.num_steps * args.batch_size, crop_size=input_size_target, set=args.set),
                                    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                    pin_memory=True)
    else:
        raise NotImplementedError('Unavailable target domain')

    targetloader_iter = enumerate(targetloader)

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    elif args.gan == 'Hinge':
        adversarial_loss = Hinge(model_D)

    # labels for adversarial training
    source_label = 1
    target_label = 0

    model.train()
    model_D.train()

    if not args.source_only and args.memory and not args.num_dataset == 1:
        ref_model = copy.deepcopy(model)  # reference model for distillation loss
        for params in ref_model.parameters():
            params.requires_grad = False

    # cudnn.benchmark = True
    optimizer.zero_grad()
    optimizer_D.zero_grad()
    if (args.warper or args.spadeWarper):
        optimizer_warp.zero_grad()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

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

        if (args.warper or args.spadeWarper):
            optimizer_warp.zero_grad()
            adjust_learning_rate(optimizer_warp, i_iter, args)

        # train G
        if not args.source_only:
            for param in model_D.parameters():
                param.requires_grad = False

        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        if (args.warper or args.spadeWarper) and args.memory:
            pred_warped, _, pred, pred_ori = model(images)
        elif args.warper:
            _, pred_warped, _, pred = model(images)
        elif args.spadeWarper:
            _, pred_warped, _, pred = model(images, labels)
        elif args.memory:
            _, _, pred, pred_ori = model(images)
        else:
            _, _, _, pred = model(images)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value_before_warped += loss_seg.item()

        if not args.source_only and args.memory and not args.num_dataset == 1:
            _, _, _, old_outputs = ref_model(images)
            loss_distillation = distillation_loss(pred_ori, old_outputs)
            loss += args.lambda_distillation * loss_distillation
            loss_distillation_value += loss_distillation.item()

        loss.backward()

        _, batch = targetloader_iter.__next__()
        images_target, _, _= batch
        images_target = images_target.to(device)

        if args.warper and args.memory:
            pred_target_warped, _, pred_target, _ = model(images_target)
        elif (args.warper or args.spadeWarper):
            _, pred_target_warped, _, pred_target = model(images_target)
        elif args.memory:
            _, _, pred_target, _ = model(images_target)
        else:
            _, _, _, pred_target = model(images_target)

        D_out = model_D(F.softmax(pred_target, dim=1))

        if args.gan == 'Hinge':
            loss_adv_target = adversarial_loss(D_out, generator=True)

        else:
            loss_adv_target = bce_loss(D_out,
                                       torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

        loss = args.lambda_adv_target[args.num_dataset - 1] * loss_adv_target
        loss_adv_target_value += loss_adv_target.item()
        loss.backward()
        optimizer.step()

        if (args.warper or args.spadeWarper):
            optimizer_warp.step()

        for k in range(args.iteration_disc):
            # train D

            if (args.warper or args.spadeWarper) and args.memory:
                pred_warped, _, pred, pred_ori = model(images)
            elif args.warper:
                _, pred_warped, _, pred = model(images)
            elif args.spadeWarper:
                _, pred_warped, _, pred = model(images, labels)
            elif args.memory:
                _, _, pred, pred_ori = model(images)
            else:
                _, _, _, pred = model(images)

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            # train with source
            pred = pred.detach()

            # train with target
            pred_target = pred_target.detach()

            if args.gan == 'Hinge':
                D_out_source = model_D(F.softmax(pred, dim=1))
                D_out_target = model_D(F.softmax(pred_target, dim=1))

                loss_D = adversarial_loss(D_out_target, D_out_source, generator=False)
                loss_D.backward()
                loss_D_value += loss_D.item()

            else:
                D_out = model_D(F.softmax(pred, dim=1))
                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                loss_D = loss_D / 2
                loss_D.backward()
                loss_D_value += loss_D.item()

                D_out = model_D(F.softmax(pred_target, dim=1))
                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
                loss_D = loss_D / 2
                loss_D.backward()
                loss_D_value += loss_D.item()

            optimizer_D.step()
        loss_D_value = loss_D_value / args.iteration_disc

        # if (args.warper or args.spadeWarper):  # retain_graph=True -> CUDA out of memory
        #     if args.memory:  # feed-forwarding the input again
        #         pred_warped, _, pred, _ = model(images)
        #     else:
        #         _, pred_warped, _, pred = model(images)
        #
        #     loss_seg = seg_loss(pred_warped, labels)
        #     loss = loss_seg
        #     loss_seg_value_after_warped += loss_seg.item()
        #     loss.backward()
        #     optimizer_warp.step()


        if i_iter % 10 == 0:
            if args.tensorboard:
                scalar_info = {
                    'Train/loss_seg_before_warped': loss_seg_value_before_warped,
                    'Train/loss_seg_after_warped': loss_seg_value_after_warped,
                    'Train/loss_distillation': loss_distillation_value,
                    'Train/loss_adv_target': loss_adv_target_value,
                    'Train/loss_D': loss_D_value
                }
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print(
            'iter = {0:8d}/{1:8d}, loss_seg_before_warped = {2:.3f} loss_seg_after_warped = {3:.3f} loss_distill = {4:.3f} loss_adv = {5:.3f} loss_D = {6:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value_before_warped, loss_seg_value_after_warped, loss_distillation_value,
                loss_adv_target_value, loss_D_value))

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            model_save_name, info = save_model(i_iter, args, model, model_D, optimizer, optimizer_D, optimizer_warp,
                                               args.snapshot_dir, args.dir_name)
            model_save_name += '_' + str(i_iter)
            torch.save(info, model_save_name + '.pth')

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            model_save_name, info = save_model(i_iter, args, model, model_D, optimizer, optimizer_D, optimizer_warp,
                                               args.snapshot_dir, args.dir_name)
            model_save_name += '_' + str(args.num_steps_stop)
            torch.save(info, model_save_name + '.pth')
            break

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
