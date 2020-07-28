import torch
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
from options import TrainOptions
from model.deeplab import Deeplab
from model.discriminator import FCDiscriminator
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.synthia_dataset import SYNTHIADataSet

PRE_TRAINED_SEG = ''
# PRE_TRAINED_SEG = './snapshots/GTA5_SYNTHIA_best.pth'

args = TrainOptions().parse()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        if args.from_scratch:
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
        else:
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr

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
    model = Deeplab(args=args)
    if args.source_only:  # training model from pre-trained ResNet on source domain
        saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
        model.load_state_dict(new_params)
    else:
        if args.from_scratch:  # training model from pre-trained ResNet on source & target domain
            saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                # Scale.layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:  # training model from pre-trained DeepLab on source & target domain
            saved_state_dict = torch.load(PRE_TRAINED_SEG, map_location=device)
            model.load_state_dict(saved_state_dict)

    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    if not args.source_only:
        model_D = FCDiscriminator(num_classes=args.num_classes).to(device)

        model_D.train()
        model_D.to(device)

    # Dataloader
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size, ignore_label=args.ignore_label, set=args.set),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    trainloader_iter = enumerate(trainloader)

    if not args.source_only:
        if args.target == 'CityScapes':
            targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.batch_size,
                                                             crop_size=input_size_target,
                                                             ignore_label=args.ignore_label,
                                                             set=args.set),
                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        elif args.target == 'SYNTHIA':
            targetloader = data.DataLoader(SYNTHIADataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.batch_size,
                                                             crop_size=input_size_target,
                                                             ignore_label=args.ignore_label,
                                                             set=args.set),
                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        else:
            raise NotImplementedError('Unavailable target domain')
        targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    if not args.source_only:
        optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D.zero_grad()

        if args.gan == 'Vanilla':
            bce_loss = torch.nn.BCEWithLogitsLoss()
        elif args.gan == 'LS':
            bce_loss = torch.nn.MSELoss()

        # labels for adversarial training
        source_label = 1
        target_label = 0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # start training
    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        loss_adv_value = 0
        loss_D_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        if not args.source_only:
            optimizer_D.zero_grad()
            adjust_learning_rate_D(optimizer_D, i_iter)

        # train G
        if not args.source_only:
            for param in model_D.parameters():
                param.requires_grad = False

        _, batch = trainloader_iter.__next__()

        images, labels, _, = batch
        images = images.to(device)
        labels = labels.long().to(device)

        pred = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value += loss_seg.item()

        loss.backward()

        if not args.source_only:
            _, batch = targetloader_iter.__next__()
            images_target, _, _ = batch
            images_target = images_target.to(device)

            pred_target = model(images_target, input_size_target)

            D_out = model_D(F.softmax(pred_target, dim=1))

            loss_adv = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv * loss_adv
            loss_adv_value += loss_adv.item()
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

        print('exp = {}'.format(args.snapshot_dir))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value, loss_adv_value, loss_D_value))

        # Snapshots directory
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        # Save model
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, str(args.num_steps_stop) + '.pth'))
            if not args.source_only:
                torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, str(i_iter) + '.pth'))
            if not args.source_only:
                torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, str(i_iter) + '_D.pth'))

if __name__ == '__main__':
    main()
