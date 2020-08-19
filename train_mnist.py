import torch
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
from options_mnist import TrainOptions
from model.deeplab import Deeplab
import torchvision.transforms as transforms
import torchvision
from model.discriminator_FC import FCDiscriminator, Hinge
import pdb
from tensorboardX import SummaryWriter
from model.AlexNet import AlexNet_DM

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

    # cudnn.enabled = True

    # Create network
    model = AlexNet_DM(num_classes=10, args=args)
    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    if not args.source_only:
        model_D = FCDiscriminator().to(device)
        model_D.train()
        model_D.to(device)
        optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D.zero_grad()

    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.0,))
    ])

    # Dataloader
    trainloader = data.DataLoader(
        torchvision.datasets.MNIST('./dataset/mnist_dataset', train=True, transform=mnist_transform, target_transform=None, download=False),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    if not args.source_only:
        targetloader = data.DataLoader(
            torchvision.datasets.MNIST('./dataset/mnist_dataset', train=True, transform=mnist_transform,
                                       target_transform=None, download=False, degree=30),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        targetloader_iter = enumerate(targetloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    if not args.source_only:
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

    classify_loss = torch.nn.CrossEntropyLoss()

    args.dir_name = args.dir_name + str(args.degree)
    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

    # start training
    for i_iter in range(args.num_steps):

        loss_classify_value = 0
        loss_adv_value = 0
        loss_D_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        if not args.source_only:
            optimizer_D.zero_grad()
            adjust_learning_rate_D(optimizer_D, i_iter)

        # train G

        for param in model_D.parameters():
            param.requires_grad = False

        _, batch = trainloader_iter.__next__()

        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        feat_new, feat_ori, pred, output_ori  = model(images)
        # pdb.set_trace()
        loss_class = classify_loss(pred, labels)
        loss = loss_class
        loss_classify_value += loss_class.item()

        loss.backward()

        _, batch = targetloader_iter.__next__()
        images_target, _ = batch
        images_target = images_target.to(device)

        if not args.source_only:

            feat_new_target, feat_ori_target, pred_target, output_ori_target = model(images_target)

            # pdb.set_trace()
            if args.gan == 'Hinge':
                loss_adv = adversarial_loss(feat_new_target, generator=True)
            else:
                D_out = model_D(feat_new_target)

                loss_adv = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

            loss = args.lambda_adv * loss_adv
            loss_adv_value += loss_adv.item()
            loss.backward()

            # train D

            # bring back requires_grad
            for param in model_D.parameters():
                param.requires_grad = True

            feat_new = feat_new.detach()
            feat_new_target = feat_new_target.detach()

            if args.gan == 'Hinge':
                # pdb.set_trace()
                loss_D = adversarial_loss(feat_new_target,feat_new, generator=False)
                loss_D.backward()
                loss_D_value += loss_D.item()
            else:
                # train with source
                # pdb.set_trace()
                D_out = model_D(feat_new)

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                loss_D = loss_D / 2

                loss_D.backward()
                loss_D_value += loss_D.item()

                # train with target
                D_out = model_D(feat_new_target)

                loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))
                loss_D = loss_D / 2

                loss_D.backward()
                loss_D_value += loss_D.item()

        optimizer.step()
        if not args.source_only:
            optimizer_D.step()

        if args.tensorboard:
            if i_iter % 10 == 0:
                writer.add_scalars('Train/loss', {'train': loss_classify_value}, i_iter)
                if not args.source_only:

                    writer.add_scalars('Train/loss_adv_target', {'train': loss_adv_value}, i_iter)
                    writer.add_scalars('Train/loss_D', {'train': loss_D_value}, i_iter)

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print(
            'iter = {0:8d}/{1:8d}, loss_classify = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f}'.format(
                i_iter, args.num_steps, loss_classify_value, loss_adv_value, loss_D_value))

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