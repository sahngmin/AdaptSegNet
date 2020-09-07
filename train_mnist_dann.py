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
from model.MNIST_model import AlexNet_DM, AlexNet_Source, DANNModel
from data.DigitFive.datasets.dataset_read import dataset_read

# PRE_TRAINED_SEG = './snapshots/GTA5_CityScapes/30000.pth'

args = TrainOptions().parse()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_power(base_lr, iter, power, interval):
    return base_lr * pow(power, int(iter / interval))


def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    # lr = lr_power(args.learning_rate, i_iter, 0.9, 1000)
    if args.from_scratch:
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr
    else:
        for i in range(len(optimizer.param_groups) - 1):
            optimizer.param_groups[i]['lr'] = lr * 0.01
        optimizer.param_groups[-1]['lr'] = lr


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    # lr = lr_power(args.learning_rate, i_iter, 0.9, 1000)
    optimizer.param_groups[0]['lr'] = lr


def main():
    args.from_scratch = False

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

    elif args.case == 2:
        args.target = 'mnistm'
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_usps' + str(args.lambda_adv) + '/6000.pth'

    elif args.case == 3:
        args.target = 'syn'
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_mnistm' + str(args.lambda_adv) + '/6000.pth'

    elif args.case == 4:
        args.target = 'svhn'
        if not args.from_scratch:
            args.pre_trained = './snapshots/mnist_syn' + str(args.lambda_adv) + '/6000.pth'

    # elif args.case == 5:
    #     args.target = 'union'
    #     if not args.from_scratch:
    #         args.pre_trained = './snapshots/mnist_syn' + str(args.lambda_adv) + '/3000.pth'


    # Create network
    model = DANNModel()
    model.train()
    model.to(device)

    if args.pre_trained is not '':
        saved_state_dict = torch.load(args.pre_trained, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                if 'domain_classifier' not in i:
                    new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)


    # targetloader = data.DataLoader(
    #     torchvision.datasets.MNIST('./dataset/mnist_dataset', train=True, transform=mnist_transform,
    #                                target_transform=None, download=False, degree=30),
    #     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)





    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.Adam(model.optim_parameters(args),lr=args.learning_rate, betas=(0.9, 0.99))
    optimizer.zero_grad()

    if not args.source_only:
        bce_loss = torch.nn.NLLLoss()

        # labels for adversarial training
        source_label = 0
        target_label = 1

    classify_loss = torch.nn.NLLLoss()

    args.dir_name = args.dir_name + args.target + str(args.lambda_adv)

    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

    # start training
    for epoch in range(2):
        # Dataloader
        trainloader, test_dataloader_source = dataset_read('mnist', args.batch_size)
        trainloader_iter = enumerate(trainloader)

        targetloader, test_dataloader_target = dataset_read(args.target, args.batch_size)
        targetloader_iter = enumerate(targetloader)

        for i_iter in range(3000):

            i_iter = epoch * 3000 + i_iter

            loss_classify_value = 0

            p = float(i_iter) / 6000
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, args)

                # train G

            _, batch = trainloader_iter.__next__()

            images, labels = batch
            images = images.to(device)
            labels = labels.long().to(device)

            class_output, domain_output = model(images, alpha)
            # pdb.set_trace()
            loss_class = classify_loss(class_output, labels)
            loss = loss_class
            loss_classify_value += loss_class.item()

            domain_label = torch.zeros(args.batch_size)

            loss_adv_S = bce_loss(domain_output, domain_label.long().to(device))
            loss += loss_adv_S

            if not args.source_only:
                _, batch = targetloader_iter.__next__()
                images_target, _ = batch
                images_target = images_target.to(device)
                class_output, domain_output = model(images_target, alpha)

                # pdb.set_trace()
                domain_label = torch.ones(args.batch_size)

                loss_adv_T = bce_loss(domain_output, domain_label.long().to(device))

                loss += loss_adv_T
            loss.backward()
            optimizer.step()

            if i_iter % 100 == 0:
                print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
                print(
                    'iter = {0:8d}/{1:8d}, loss_classify = {2:.3f}'.format(
                        i_iter, args.num_steps, loss_classify_value))

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
