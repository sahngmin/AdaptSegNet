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
from model.discriminator import FCDiscriminator, SpectralDiscriminator, Hinge
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from tensorboardX import SummaryWriter

PRE_TRAINED_SEG = ''
# PRE_TRAINED_SEG = './snapshots/GTA5_CityScapes/30000.pth'

args = TrainOptions().parse()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_power(base_lr, iter, power, interval):
    return base_lr * pow(power, int(iter / interval))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    # lr = lr_power(args.learning_rate, i_iter, 0.9, 1000)
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
    # lr = lr_power(args.learning_rate, i_iter, 0.9, 1000)
    optimizer.param_groups[0]['lr'] = lr

def main():
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 gpu 연산 무작위 고정
    torch.backends.cudnn.enabled = True  # cudnn library를 사용하지 않게 만듬
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    # cudnn.enabled = True

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
        if args.gan == 'LS' or 'Vanilla':
            model_D = FCDiscriminator(num_classes=args.num_classes).to(device)
        elif args.gan == 'Hinge':
            model_D = SpectralDiscriminator(num_classes=args.num_classes).to(device)
            # model_D = FCDiscriminator(num_classes=args.num_classes).to(device)
        else:
            raise NotImplementedError('Unavailable GAN option')

        model_D.train()
        model_D.to(device)

    # Dataloader
    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                        crop_size=input_size, ignore_label=args.ignore_label,
                        set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    elif args.source == 'SYNTHIA':
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                           crop_size=input_size, ignore_label=args.ignore_label,
                           set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError('Unavailable source domain')
    trainloader_iter = enumerate(trainloader)

    if not args.source_only:
        if args.target == 'CityScapes':
            targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.batch_size,
                                                             crop_size=input_size_target,
                                                             ignore_label=args.ignore_label,
                                                             set=args.set, num_classes=args.num_classes),
                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)
        elif args.target == 'IDD':
            targetloader = data.DataLoader(IDDDataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.batch_size,
                                                             crop_size=input_size_target,
                                                             ignore_label=args.ignore_label,
                                                             set=args.set, num_classes=args.num_classes),
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
        elif args.gan == 'Hinge':
            adversarial_loss = Hinge(model_D)
        else:
            raise NotImplementedError('Unavailable GAN option')

        # labels for adversarial training
        source_label = 1
        target_label = 0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

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

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        pred = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value += loss_seg.item()

        if not args.new_hinge:
            loss.backward()

        if not args.source_only:
            _, batch = targetloader_iter.__next__()
            images_target, _, _ = batch
            images_target = images_target.to(device)

            pred_target = model(images_target, input_size_target)

            if args.gan == 'Hinge':
                if args.new_hinge:
                    source_pred = F.softmax(pred, dim=1)
                else:
                    source_pred = None
                loss_adv = adversarial_loss(F.softmax(pred_target, dim=1), real_samples=source_pred, generator=True, new_hinge=args.new_hinge)
            else:
                D_out = model_D(F.softmax(pred_target, dim=1))

                loss_adv = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss_G = args.lambda_adv * loss_adv
            loss_adv_value += loss_adv.item()

            if args.new_hinge:
                loss += loss_G
                loss.backward()
            else:
                loss_G.backward()

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            pred = pred.detach()
            pred_target = pred_target.detach()

            if args.gan == 'Hinge':
                loss_D = adversarial_loss(F.softmax(pred_target, dim=1), F.softmax(pred, dim=1), generator=False, new_hinge=args.new_hinge)
                loss_D.backward()
                loss_D_value += loss_D.item()
            else:
                # train with source
                D_out = model_D(F.softmax(pred, dim=1))

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                loss_D = loss_D / 2

                loss_D.backward()
                loss_D_value += loss_D.item()

                # train with target
                D_out = model_D(F.softmax(pred_target, dim=1))

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
                loss_D = loss_D / 2

                loss_D.backward()
                loss_D_value += loss_D.item()

        optimizer.step()
        if not args.source_only:
            optimizer_D.step()


        if args.tensorboard:
            scalar_info = {
                'Train/loss_seg': loss_seg_value,
                'Train/loss_adv_target': loss_adv_value,
                'Train/loss_D': loss_D_value
            }

            if i_iter % 10 == 0:
                for key, val in scalar_info.items():
                    writer.add_scalar(key, val, i_iter)

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value, loss_adv_value, loss_D_value))

        # Snapshots directory
        if not os.path.exists(osp.join(args.snapshot_dir, args.dir_name)):
            os.makedirs(osp.join(args.snapshot_dir, args.dir_name))

        # Save model
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '.pth'))
            if not args.source_only:
                torch.save(model_D.state_dict(),
                           osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))
            if not args.source_only:
                torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '_D.pth'))

        if args.tensorboard:
            writer.close()


if __name__ == '__main__':
    main()