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
from model.MNIST_model import AlexNet_DM, AlexNet_Source
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
    for i in range(len(optimizer.param_groups)):
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
    # Create network
    model = AlexNet_Source(num_classes=10)
    model.train()
    model.to(device)

    if args.pre_trained is not '':
        saved_state_dict = torch.load(args.pre_trained, map_location=device)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

    args.dir_name = 'svhn'
    # Dataloader
    trainloader, test_dataloader_source = dataset_read(args.dir_name, args.batch_size)
    trainloader_iter = enumerate(trainloader)


    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    classify_loss = torch.nn.NLLLoss()


    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        writer = SummaryWriter(os.path.join(args.log_dir, args.dir_name))

    # start training
    for i_iter in range(args.num_steps):

        loss_classify_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        _, batch = trainloader_iter.__next__()

        images, labels = batch
        images = images.to(device)
        labels = labels.long().to(device)

        feat_new, feat_ori, pred, output_ori = model(images)
        # pdb.set_trace()
        loss_class = classify_loss(pred, labels)
        loss = loss_class
        loss_classify_value += loss_class.item()
        loss.backward()
        optimizer.step()

        if args.tensorboard:
            if i_iter % 10 == 0:
                writer.add_scalars('Train/loss', {'train': loss_classify_value}, i_iter)

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

            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()