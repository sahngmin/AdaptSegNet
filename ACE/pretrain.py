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
import argparse
from model.deeplab import Deeplab
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.gta5_dataset import GTA5DataSet

SOURCE = 'SYNTHIA'  # 'GTA5' or 'SYNTHIA'
DATA_DIRECTORY = '/work/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
# DATA_DIRECTORY = '/work/SYNTHIA'
# DATA_LIST_PATH = './dataset/synthia_list/train.txt'
RESTORE_FROM_RESNET = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 10

def get_arguments():
    parser = argparse.ArgumentParser(description="ACE pre-training")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--restore-from-resnet", type=str, default=RESTORE_FROM_RESNET,
                        help="Where restore model parameters from.")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--set", type=str, default='train')
    parser.add_argument("--save-pred-every", type=int, default=5000)
    parser.add_argument("--num-steps-stop", type=int, default=30000)
    parser.add_argument("--num-steps", type=int, default=30000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)

    parser.add_argument("--random-seed", type=int, default=1338)

    return parser.parse_args()

def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    input_size = (512, 256)

    cudnn.enabled = True

    # Create and load network
    model = Deeplab(args=args)
    saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                    crop_size=input_size, ignore_label=args.ignore_label,
                    set=args.set, num_classes=args.num_classes),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # trainloader = data.DataLoader(
    #     SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
    #                    crop_size=input_size, ignore_label=args.ignore_label,
    #                    set=args.set, num_classes=args.num_classes),
    #     batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    model.train()
    model.to(device)
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)

    cudnn.benchmark = True
    optimizer.zero_grad()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    for i_iter in range(args.num_steps):

        loss_seg_value = 0
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        pred = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value += loss_seg.item()

        loss.backward()
        optimizer.step()

        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(i_iter, args.num_steps, loss_seg))

        if not os.path.exists('./pretrain_snapshots'):
            os.makedirs('./pretrain_snapshots')

        if i_iter >= args.num_steps_stop - 1:
            torch.save(model.state_dict(), osp.join('./pretrain_snapshots', 'pretrain_' + args.source + '_'
                                                    + str(args.num_steps_stop) + '.pth'))
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            torch.save(model.state_dict(), osp.join('./pretrain_snapshots', 'pretrain_' + args.source + '_'
                                                    + str(i_iter) + '.pth'))

if __name__ == '__main__':
    main()