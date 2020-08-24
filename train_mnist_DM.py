import torch
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
import copy

from options_mnist import TrainOptions
from model.deeplab import Deeplab
import torchvision.transforms as transforms
import torchvision
from model.discriminator_FC import FCDiscriminator, Hinge
import pdb
from tensorboardX import SummaryWriter
from model.MNIST_model import AlexNet_DM, AlexNet_Source
from data.DigitFive.datasets.dataset_read import dataset_read

# PRE_TRAINED_SEG = './snapshots/GTA5_CityScapes/30000.pth'

args = TrainOptions().parse()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_power(base_lr, iter, power, interval):
    return base_lr * pow(power, int(iter / interval))

def distillation_loss(pred_origin, old_outputs):
    pred_origin_logsoftmax = (pred_origin / 2).log_softmax(dim=1)
    old_outputs = (old_outputs / 2).softmax(dim=1)
    loss_distillation = (-(old_outputs * pred_origin_logsoftmax)).sum(dim=1)
    loss_distillation = loss_distillation.sum() / loss_distillation.flatten().shape[0]
    return loss_distillation

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
    cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # cudnn.enabled = True
    continual_list = ['mnist', 'usps', 'mnistm', 'syn', 'svhn']

    if args.case == 1:
        args.target = 'usps'
        args.lambda_adv = args.lambda_adv
        num_target = 1


    elif args.case == 2:
        args.target = 'mnistm'
        num_target = 2
        args.lambda_adv = args.lambda_adv
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_usps' + str(args.lambda_adv) + '/3000.pth'

    elif args.case == 3:
        args.target = 'syn'
        num_target = 3
        args.lambda_adv = args.lambda_adv
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_mnistm' + str(args.lambda_adv) + '/3000.pth'

    elif args.case == 4:
        args.target = 'svhn'
        num_target = 4
        args.lambda_adv = args.lambda_adv
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_syn' + str(args.lambda_adv) + '/3000.pth'

    else:
        num_target = continual_list.index(args.target)

    # Create network
    model = AlexNet_DM(num_classes=10, num_target=num_target)
    model.train()
    model.to(device)

    if not args.from_scratch:
        saved_state_dict = torch.load(args.pre_trained, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        ref_model = copy.deepcopy(model)  # reference model for knowledge distillation
        for params in ref_model.parameters():
            params.requires_grad = False
        ref_model.eval()

    # init D
    if not args.source_only:
        model_D = FCDiscriminator().to(device)
        model_D.train()
        model_D.to(device)
        optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_D.zero_grad()

    # Dataloader
    trainloader, test_dataloader_source = dataset_read('mnist', args.batch_size)
    trainloader_iter = enumerate(trainloader)

    if not args.source_only:
        targetloader, test_dataloader_target = dataset_read(args.target, args.batch_size)
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

    args.dir_name = args.dir_name + args.target + str(args.lambda_adv)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

    # start training
    for i_iter in range(args.num_steps):

        loss_classify_value = 0
        loss_distill_value = 0
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

        feat_new, feat_ori, pred, output_ori = model(images)
        # pdb.set_trace()
        loss_class = classify_loss(pred, labels)
        loss = loss_class
        loss_classify_value += loss_class.item()

        if not args.from_scratch:
            _, old_outputs = ref_model(images)
            loss_distill = distillation_loss(output_ori, old_outputs)
            loss += args.lambda_distill * loss_distill
            loss_distill_value += loss_distill.item()

        loss.backward()

        if not args.source_only:
            _, batch = targetloader_iter.__next__()
            images_target, _ = batch
            images_target = images_target.to(device)
            feat_new_target, feat_ori_target, pred_target, output_ori_target = model(images_target)

            # pdb.set_trace()
            if args.gan == 'Hinge':
                feat_new = feat_new.detach()
                loss_adv = adversarial_loss(feat_new_target, feat_new, generator=True, new_Hinge=True)
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
                loss_D = adversarial_loss(feat_new_target, feat_new, generator=False)
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
            # if not args.source_only:
            #     torch.save(model_D.state_dict(),
            #                osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '_D.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))
            # if not args.source_only:
            #     torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '_D.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
