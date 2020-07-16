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
from dataset.synthia_dataset import SYNTHIADataSet
from compute_iou import fast_hist, per_class_iu

PRE_TRAINED_SEG = ''
# PRE_TRAINED_DISC = ''
PRE_TRAINED_DISC = None

IMG_MEAN = np.array((0, 0, 0), dtype=np.float32)

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
        model_D.to(device)
        optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

        targetloader = torch.utils.data.DataLoader(SYNTHIADataSet(args.data_dir_target, args.data_list_target,
                                                      max_iters=args.num_steps * args.batch_size,
                                                      crop_size=input_size_target,
                                                      scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                      set=args.set),
                                       batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                       pin_memory=True)
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

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    # Dataloader
    trainloader = torch.utils.data.DataLoader(SYNTHIADataSet(args.data_dir, args.data_list,
                                                 max_iters=args.num_steps * args.batch_size,
                                                 crop_size=input_size,
                                                 scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                                  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
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
            images_target, _, _, _ = batch
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

        # Snapshots directory
        if args.source_only:
            if args.warper and args.memory:
                if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_warp_DM')):
                    os.makedirs(osp.join(args.snapshot_dir, 'source_only_warp_DM'))
            elif args.warper:
                if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_warp')):
                    os.makedirs(osp.join(args.snapshot_dir, 'source_only_warp'))
            elif args.memory:
                if not os.path.exists(osp.join(args.snapshot_dir, 'source_only_DM')):
                    os.makedirs(osp.join(args.snapshot_dir, 'source_only_DM'))
            else:
                if not os.path.exists(osp.join(args.snapshot_dir, 'source_only')):
                    os.makedirs(osp.join(args.snapshot_dir, 'source_only'))
        else:
            if args.warper and args.memory:
                if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_warp_DM')):
                    os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_warp_DM'))
            elif args.warper:
                if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_warp')):
                    os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_warp'))
            elif args.memory:
                if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment_DM')):
                    os.makedirs(osp.join(args.snapshot_dir, 'single_alignment_DM'))
            else:
                if not os.path.exists(osp.join(args.snapshot_dir, 'single_alignment')):
                    os.makedirs(osp.join(args.snapshot_dir, 'single_alignment'))

        # ---------------------------------------------- save model -----------------------------------------------------
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            if args.source_only:
                if args.warper and args.memory:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp_DM',
                                                            str(args.source) + '_' + str(args.num_steps_stop) + '.pth'))
                elif args.warper:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp',
                                                            str(args.source) + '_' + str(args.num_steps_stop) + '.pth'))
                elif args.memory:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_DM',
                                                            str(args.source) + '_' + str(args.num_steps_stop) + '.pth'))
                else:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only',
                                                            str(args.source) + '_' + str(args.num_steps_stop) + '.pth'))
            else:
                if args.num_dataset == 1:
                    if args.warper and args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    elif args.warper:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    elif args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    else:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                else:
                    targetlist = list(dataset_dict.keys())
                    filename = str(args.source) + 'to'
                    for i in range(args.num_dataset - 1):
                        filename += targetlist[i]
                        filename += 'to'
                    if args.warper and args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    elif args.warper:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    elif args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
                    else:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            filename + str(args.target) + '_' + str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('checking validation accuracy ...')
            train_val_loader = torch.utils.data.DataLoader(SYNTHIADataSet(args.data_dir, './dataset/synthia_seqs_04_spring_list/val.txt',
                                                         max_iters=None,
                                                         crop_size=input_size,
                                                         scale=False, mirror=False,
                                                         mean=IMG_MEAN, set='val'),
                                          batch_size=args.batch_size, shuffle=False, pin_memory=True)
            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(train_val_loader):
                images_val, labels, _, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.warper and args.memory:
                    pred_warped, _, pred, _ = model(images_val, input_size)
                elif args.warper:
                    _, pred_warped, _, pred = model(images_val, input_size)
                elif args.memory:
                    _, _, pred, _ = model(images_val, input_size)
                else:
                    _, _, _, pred = model(images_val, input_size)
                labels = labels.squeeze()
                pred = nn.Upsample(size=(760, 1280), mode='bilinear', align_corners=True)(pred)
                _, pred = pred.squeeze().max(dim=0)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            print('===> mIoU (Source): ' + str(round(np.nanmean(mIoUs) * 100, 2)))

            val_val_loader = torch.utils.data.DataLoader(
                SYNTHIADataSet(args.data_dir_target, './dataset/synthia_seqs_02_spring_list/val.txt',
                               max_iters=None,
                               crop_size=input_size,
                               scale=False, mirror=False,
                               mean=IMG_MEAN, set='val'),
                batch_size=args.batch_size, shuffle=False, pin_memory=True)
            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(val_val_loader):
                images_val, labels, _, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.warper and args.memory:
                    pred_warped, _, pred, _ = model(images_val, input_size)
                elif args.warper:
                    _, pred_warped, _, pred = model(images_val, input_size)
                elif args.memory:
                    _, _, pred, _ = model(images_val, input_size)
                else:
                    _, _, _, pred = model(images_val, input_size)
                labels = labels.squeeze()
                pred = nn.Upsample(size=(760, 1280), mode='bilinear', align_corners=True)(pred)
                _, pred = pred.squeeze().max(dim=0)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            print('===> mIoU (Target): ' + str(round(np.nanmean(mIoUs) * 100, 2)))

            print('taking snapshot ...')
            if args.source_only:
                if args.warper and args.memory:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp_DM',
                                                            str(args.source) + '_' + str(i_iter) + '.pth'))
                elif args.warper:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_warp',
                                                            str(args.source) + '_' + str(i_iter) + '.pth'))
                elif args.memory:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only_DM',
                                                            str(args.source) + '_' + str(i_iter) + '.pth'))
                else:
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only',
                                                            str(args.source) + '_' + str(i_iter) + '.pth'))
            else:
                if args.num_dataset == 1:
                    if args.warper and args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    elif args.warper:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    elif args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    else:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            str(args.source) + 'to' + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                else:
                    targetlist = list(dataset_dict.keys())
                    filename = 'GTA5to'
                    for i in range(args.num_dataset - 1):
                        filename += targetlist[i]
                        filename += 'to'
                    if args.warper and args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            filename + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp_DM',
                                            filename + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    elif args.warper:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            filename + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_warp',
                                            filename + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    elif args.memory:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            filename + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment_DM',
                                            filename + str(args.target) + '_' + str(i_iter) + '_D.pth'))
                    else:
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            filename + str(args.target) + '_' + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_alignment',
                                            filename + str(args.target) + '_' + str(i_iter) + '_D.pth'))


    if args.tensorboard:
        writer.close()

if __name__ == '__main__':
    main()
