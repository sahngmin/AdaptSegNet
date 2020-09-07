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

from model.discriminator_FC import FCDiscriminator, Hinge, FCDiscriminator_Spec
import pdb
from tensorboardX import SummaryWriter
from model.MNIST_model import AlexNet_DM
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
    if args.from_scratch:
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

    else:
        for i in range(len(optimizer.param_groups)-1):
            optimizer.param_groups[i]['lr'] = 0.0 * lr
        optimizer.param_groups[-1]['lr'] = lr
        # optimizer.param_groups[-2]['lr'] = 10 * lr


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    # lr = lr_power(args.learning_rate, i_iter, 0.9, 1000)
    optimizer.param_groups[0]['lr'] = lr


def main():
    args.from_scratch = False
    args.learning_rate = 1e-3
    args.learning_rate_D = 1e-3

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

    load_path = ''

    if args.case == 1:
        args.target = continual_list[1]
        num_target = 1
        args.learning_rate = 2.5e-4
        args.learning_rate_D = 2e-4
        # load_path = './snapshots/mnist/3000.pth'
        if not args.from_scratch:
            args.lambda_adv = 1.3
            # load_path = './snapshots/mnist/3000.pth'

    elif args.case == 2:
        args.target = continual_list[2]
        num_target = 2
        if not args.from_scratch:
            args.lambda_adv = 3.0
            load_path = './snapshots/mnist_usps' + '5.0' + 'Hinge_87.31' + '/6000.pth'
            # load_path = './snapshots/mnist_usps' + str(args.lambda_adv) + 'Hinge' + '/5400.pth'

    elif args.case == 3:
        args.target = continual_list[3]
        num_target = 3
        if not args.from_scratch:
            args.lambda_adv = 2.0
            load_path = './snapshots/mnist_mnistm' + '3.0' + 'Hinge' + '/6000.pth'
            # load_path = './snapshots/mnist_mnistm' + str(args.lambda_adv) + 'Hinge' + '/6000.pth'

    elif args.case == 4:
        args.target = continual_list[4]
        num_target = 4
        if not args.from_scratch:
            args.lambda_adv = 3.0
            load_path = './snapshots/mnist_syn' + '2.0' + 'Hinge' + '/6000.pth'
            # load_path = './snapshots/mnist_syn' + str(args.lambda_adv) + 'Hinge' + '/6000.pth'

    else:
        num_target = continual_list.index(args.target)

    # Create network
    model = AlexNet_DM(num_classes=10, num_target=num_target)
    model.train()
    model.to(device)

    if load_path is not '':
        saved_state_dict = torch.load(load_path, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                if 'DM' in i:
                    new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)


    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.Adam(model.optim_parameters(args), lr=args.learning_rate, betas=(0.9, 0.99))

    # init D
    model_D = FCDiscriminator().to(device)
    model_D.train()
    model_D.to(device)
    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

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

    args.dir_name = args.dir_name + args.target + str(args.lambda_adv) + args.gan

    if not args.from_scratch:
        ref_model = copy.deepcopy(model)  # reference model for knowledge distillation
        for params in ref_model.parameters():
            params.requires_grad = False
        ref_model.eval()

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

    print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))

    for epoch in range(2):
        # Dataloader
        trainloader, test_dataloader_source = dataset_read('mnist', args.batch_size)
        trainloader_iter = enumerate(trainloader)

        targetloader, test_dataloader_target = dataset_read(args.target, args.batch_size)
        targetloader_iter = enumerate(targetloader)
        # start training
        for i_iter in range(3000):
            i_iter = epoch * 3000 + i_iter
            loss_classify_value = 0
            loss_distill_value = 0.0
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
            labels = labels.to(device).long()
            feat_new, feat_ori, pred, output_ori = model(images)
            loss_class = classify_loss(pred, labels)
            loss = loss_class
            loss_classify_value += loss_class.item()
            # loss.backward()

            _, _, _, old_outputs = ref_model(images)
            loss_distill = distillation_loss(output_ori, old_outputs)
            loss += args.lambda_distill * loss_distill
            loss_distill_value += loss_distill.item()

            loss.backward()

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
                loss_D = adversarial_loss(feat_new_target, feat_new, generator=False)
                loss_D.backward()
                loss_D_value += loss_D.item()
            else:
                # train with source
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

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if not args.source_only:
                torch.nn.utils.clip_grad_norm_(model_D.parameters(), 0.5)
                optimizer_D.step()

            if args.tensorboard:
                if i_iter % 20 == 0:
                    writer.add_scalars('Train/loss', {'train': loss_classify_value}, i_iter)
                    if not args.source_only:
                        writer.add_scalars('Train/loss_adv_target', {'train': loss_adv_value}, i_iter)
                        writer.add_scalars('Train/loss_D', {'train': loss_D_value}, i_iter)
            if i_iter % 100 == 0:
                print('iter = {0:8d}/{1:8d}, loss_classify = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f} '.format(
                        i_iter, args.num_steps, loss_classify_value, loss_adv_value, loss_D_value))
                print("Loss Distillation: ", loss_distill_value)

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
