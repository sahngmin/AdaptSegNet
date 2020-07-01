import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import copy

from model.deeplab_DM import Deeplab_DM
from model.deeplab import Deeplab
from model.discriminator import FCDiscriminator
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet

SOURCE_ONLY = False
MEMORY = True
SCALE = False
FROM_SCRATCH = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 300000  # early stopping
NUM_STEPS = 300000

dataset_dict = {'CityScapes': 1, 'Synthia': 2}
TARGET = 'CityScapes'
SET = 'train'
NUM_DATASET = dataset_dict[TARGET]

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9

LEARNING_RATE_D = 1e-4

GAN = 'LS'

LAMBDA_ADV_TARGET = [0.002, 0.003]
LAMBDA_ADV_MEMORY = [0.0015, 0.003]
LAMBDA_DISTILLATION = 0.1
LAMBDA_MEMORY = [1.0]
ALPHA = [0.25, 0.5]

RANDOM_SEED = 1338

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/joonhkim/UDA/datasets/GTA5'
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1024,512'
DATA_DIRECTORY_TARGET = '/home/joonhkim/UDA/datasets/CityScapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '1024,512'

NUM_CLASSES = 19

RESTORE_FROM_RESNET = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM_DEEPLAB = './snapshots/source_only/GTA5_best_model.pth'
RESTORE_FROM_PREVDOMAIN = './snapshots/single_level_DM/GTA5toCityScapes_best_model.pth'

SAVE_NUM_IMAGES = 2

SNAPSHOT_DIR = './snapshots/'
LOG_DIR = './log'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : CityScapes, Synthia")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-adv-target", type=list, default=LAMBDA_ADV_TARGET,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-memory", type=list, default=LAMBDA_ADV_MEMORY,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-memory", type=list, default=LAMBDA_MEMORY)
    parser.add_argument("--lambda-distillation", type=float, default=LAMBDA_DISTILLATION)
    parser.add_argument("--alpha", type=list, default=ALPHA)
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from-resnet", type=str, default=RESTORE_FROM_RESNET,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-deeplab", type=str, default=RESTORE_FROM_DEEPLAB,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-prevdomain", type=str, default=RESTORE_FROM_PREVDOMAIN,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument("--memory", action='store_true', default=MEMORY)
    parser.add_argument("--scale", action='store_true', default=SCALE)
    parser.add_argument("--from-scratch", action='store_true', default=FROM_SCRATCH)
    parser.add_argument("--num-dataset", type=int, default=NUM_DATASET, help="Which target dataset?")
    return parser.parse_args()


args = get_arguments()


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

    """Create the model and start the training."""

    # device = torch.device("cuda" if not args.cpu else "cpu")
    device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    if SOURCE_ONLY:  # training model from pre-trained ResNet on source domain(GTA5)
        model = Deeplab(num_classes=args.num_classes)
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

        model.train()
        model.to(device)

        cudnn.benchmark = True

        if not os.path.exists(osp.join(args.snapshot_dir, 'source_only')):
            os.makedirs(osp.join(args.snapshot_dir, 'source_only'))

        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size,
                        scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

        trainloader_iter = enumerate(trainloader)

        # implement model.optim_parameters(args) to handle different models' lr setting

        optimizer = optim.SGD(model.optim_parameters(args, SOURCE_ONLY),
                              lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer.zero_grad()

        seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

        # set up tensor board
        if args.tensorboard:
            if not os.path.exists(args.log_dir):
                os.makedirs(args.log_dir)

            writer = SummaryWriter(args.log_dir)

        for i_iter in range(args.num_steps):

            loss_seg_value = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter, args)

            for sub_i in range(args.iter_size):

                # train G with source

                _, batch = trainloader_iter.__next__()

                images, labels, _, _ = batch
                images = images.to(device)
                labels = labels.long().to(device)

                pred = model(images, input_size)

                loss_seg = seg_loss(pred, labels)
                loss = loss_seg

                # proper normalization
                loss = loss / args.iter_size
                loss.backward()
                loss_seg_value += loss_seg.item() / args.iter_size

            optimizer.step()

            if args.tensorboard:
                scalar_info = {
                    'loss_seg': loss_seg_value
                    }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.snapshot_dir))
            print("iter = {0:8d}/{1:8d}, loss_seg2 = {2:.3f}".format(i_iter, args.num_steps, loss_seg_value))

            if i_iter >= args.num_steps_stop - 1:
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only', 'GTA5_' + str(args.num_steps_stop) + '.pth'))
                break

            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                print('taking snapshot ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'source_only', 'GTA5_' + str(i_iter) + '.pth'))

        if args.tensorboard:
            writer.close()
    else:
        if not args.memory:  # single-level alignment without DM
            model = Deeplab(num_classes=args.num_classes)
            if args.from_scratch:  # load pretrained ResNet
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
            else:  # load DeepLab trained on source domain
                if args.restore_from_deeplab[:4] == 'http':
                    saved_state_dict = model_zoo.load_url(args.restore_from_deeplab)
                else:
                    saved_state_dict = torch.load(args.restore_from_deeplab)

                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    # layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    if not i_parts[0] == 'layer5':
                        new_params[i] = saved_state_dict[i]
                model.load_state_dict(new_params)

            model.train()
            model.to(device)

            cudnn.benchmark = True

            # init D
            model_D = FCDiscriminator(num_classes=args.num_classes).to(device)
            model_D.train()
            model_D.to(device)

            if not os.path.exists(osp.join(args.snapshot_dir, 'single_level')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_level'))

            trainloader = data.DataLoader(
                GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                            crop_size=input_size,
                            scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

            trainloader_iter = enumerate(trainloader)

            targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                             crop_size=input_size_target,
                                                             scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                             set=args.set),
                                           batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                           pin_memory=True)

            targetloader_iter = enumerate(targetloader)

            # implement model.optim_parameters(args) to handle different models' lr setting

            optimizer = optim.SGD(model.optim_parameters(args, SOURCE_ONLY),
                                  lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
            optimizer.zero_grad()

            optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
            optimizer_D.zero_grad()

            if args.gan == 'Vanilla':
                bce_loss = torch.nn.BCEWithLogitsLoss()
            elif args.gan == 'LS':
                bce_loss = torch.nn.MSELoss()
            seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

            # labels for adversarial training
            source_label = 0
            target_label = 1

            # set up tensor board
            if args.tensorboard:
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir)

                writer = SummaryWriter(args.log_dir)

            for i_iter in range(args.num_steps):

                loss_seg_value = 0
                loss_adv_target_value = 0
                loss_D_value = 0

                optimizer.zero_grad()
                adjust_learning_rate(optimizer, i_iter, args)

                optimizer_D.zero_grad()
                adjust_learning_rate_D(optimizer_D, i_iter)

                for sub_i in range(args.iter_size):

                    # train G

                    # don't accumulate grads in D
                    for param in model_D.parameters():
                        param.requires_grad = False

                    # train with source

                    _, batch = trainloader_iter.__next__()

                    images, labels, _, _ = batch
                    images = images.to(device)
                    labels = labels.long().to(device)

                    pred = model(images, input_size)

                    loss_seg = seg_loss(pred, labels)
                    loss = loss_seg

                    # proper normalization
                    loss = loss / args.iter_size
                    loss.backward()
                    loss_seg_value += loss_seg.item() / args.iter_size

                    # train with target

                    _, batch = targetloader_iter.__next__()
                    images, _, _ = batch
                    images = images.to(device)

                    pred_target = model(images, input_size_target)

                    D_out = model_D(F.softmax(pred_target, dim=1))

                    loss_adv_target = bce_loss(D_out,
                                                torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))

                    loss = args.lambda_adv_target[0] * loss_adv_target
                    loss = loss / args.iter_size
                    loss.backward()
                    loss_adv_target_value += loss_adv_target.item() / args.iter_size

                    # train D

                    # bring back requires_grad
                    for param in model_D.parameters():
                        param.requires_grad = True

                    # train with source
                    pred = pred.detach()

                    D_out = model_D(F.softmax(pred, dim=1))

                    loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                    loss_D = loss_D / args.iter_size / 2

                    loss_D.backward()

                    loss_D_value += loss_D.item()

                    # train with target
                    pred_target = pred_target.detach()

                    D_out = model_D(F.softmax(pred_target, dim=1))

                    loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))

                    loss_D = loss_D / args.iter_size / 2

                    loss_D.backward()
                    loss_D_value += loss_D.item()

                optimizer.step()
                optimizer_D.step()

                if args.tensorboard:
                    scalar_info = {
                        'loss_seg': loss_seg_value,
                        'loss_adv_target': loss_adv_target_value,
                        'loss_D': loss_D_value
                    }

                    if i_iter % 10 == 0:
                        for key, val in scalar_info.items():
                            writer.add_scalar(key, val, i_iter)

                print('exp = {}'.format(args.snapshot_dir))
                print(
                    'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_adv = {3:.3f} loss_D = {4:.3f}'.format(
                        i_iter, args.num_steps, loss_seg_value, loss_adv_target_value, loss_D_value))

                if i_iter >= args.num_steps_stop - 1:
                    print('save model ...')
                    torch.save(model.state_dict(),
                               osp.join(args.snapshot_dir, 'single_level', 'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                    torch.save(model_D.state_dict(),
                               osp.join(args.snapshot_dir, 'single_level', 'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                    break

                if i_iter % args.save_pred_every == 0 and i_iter != 0:
                    print('taking snapshot ...')
                    torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'single_level', 'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                    torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'single_level', 'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))

            if args.tensorboard:
                writer.close()

        else:  # single-level alignment with DM
            model = Deeplab_DM(num_classes=args.num_classes, len_dataset=args.num_dataset, args=args)

            if args.num_dataset == 1:  # first domain
                if args.from_scratch:  # load pretrained ResNet
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
                else:  # load DeepLab trained on source domain
                    saved_state_dict = torch.load(args.restore_from_deeplab)
                    new_params = model.state_dict().copy()
                    for i in saved_state_dict:
                        # layer5.conv2d_list.3.weight
                        i_parts = i.split('.')
                        if not i_parts[0] == 'layer5':
                            new_params[i] = saved_state_dict[i]
                    model.load_state_dict(new_params)

            else:  # after first domain
                saved_state_dict = torch.load(args.restore_from_prevdomain)
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    # load conv1, bn1, relu, maxpool, layer1,2,3,4,6 and DM, scale for previous tasks
                    new_params[i] = saved_state_dict[i]
                model.load_state_dict(new_params)

            model.train()
            model.to(device)

            if not args.num_dataset == 1:
                ref_model = copy.deepcopy(model)  # reference model for distillation loss
                for params in ref_model.parameters():
                    params.requires_grad = False

            cudnn.benchmark = True

            # init D
            model_D = FCDiscriminator(num_classes=args.num_classes).to(device)
            model_D.train()
            model_D.to(device)

            if not os.path.exists(osp.join(args.snapshot_dir, 'single_level_DM')):
                os.makedirs(osp.join(args.snapshot_dir, 'single_level_DM'))

            trainloader = data.DataLoader(
                GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                            crop_size=input_size,
                            scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

            trainloader_iter = enumerate(trainloader)

            if args.target == 'CityScapes':
                targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                                 max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                                 crop_size=input_size_target,
                                                                 scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
                                                                 set=args.set),
                                               batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               pin_memory=True)
            elif args.target == 'Synthia':  # SYNTHIA dataloader 필요!!!
                targetloader = data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                                 max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                                 crop_size=input_size_target,
                                                                 scale=False, mirror=args.random_mirror, mean=IMG_MEAN,
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

            optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
            optimizer_D.zero_grad()

            if args.gan == 'Vanilla':
                bce_loss = torch.nn.BCEWithLogitsLoss()
            elif args.gan == 'LS':
                bce_loss = torch.nn.MSELoss()
            seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)

            # labels for adversarial training
            source_label = 0
            target_label = 1

            # set up tensor board
            if args.tensorboard:
                if not os.path.exists(args.log_dir):
                    os.makedirs(args.log_dir)

                writer = SummaryWriter(args.log_dir)

            for i_iter in range(args.num_steps):

                loss_seg_value = 0
                loss_memory_value = 0
                loss_distillation_value = 0
                loss_adv_target_value = 0
                loss_adv_memory_value = 0
                loss_D_value = 0

                optimizer.zero_grad()
                adjust_learning_rate(optimizer, i_iter, args)

                optimizer_D.zero_grad()
                adjust_learning_rate_D(optimizer_D, i_iter)

                for sub_i in range(args.iter_size):

                    # train G

                    # don't accumulate grads in D
                    for param in model_D.parameters():
                        param.requires_grad = False

                    # train with source

                    _, batch = trainloader_iter.__next__()

                    images, labels, _, _ = batch
                    images = images.to(device)
                    labels = labels.long().to(device)

                    pred_both, pred_origin, pred_DM = model(images, input_size, args)

                    loss_seg = seg_loss(pred_both, labels)
                    loss = loss_seg
                    loss_seg_value += loss_seg.item() / args.iter_size

                    if not args.num_dataset == 1:
                        _, old_outputs, _ = ref_model(images, input_size, args)
                        loss_distillation = distillation_loss(pred_origin, old_outputs)
                        loss_memory = seg_loss(pred_DM, labels)
                        loss += args.lambda_distillation * loss_distillation + args.lambda_memory[args.num_dataset - 2] * loss_memory

                        loss_distillation_value += loss_distillation.item() / args.iter_size
                        loss_memory_value += loss_memory.item() / args.iter_size

                    loss = loss / args.iter_size
                    loss.backward()

                    # train with target

                    _, batch = targetloader_iter.__next__()
                    images, _, _ = batch
                    images = images.to(device)

                    pred_both_target, _, pred_DM_target = model(images, input_size_target, args)

                    D_out = model_D(F.softmax(pred_both_target, dim=1))
                    D_out_memory = model_D(F.softmax(pred_DM_target, dim=1))

                    loss_adv_target = bce_loss(D_out,
                                               torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                    loss_adv_memory = bce_loss(D_out_memory,
                                               torch.FloatTensor(D_out_memory.data.size()).fill_(source_label).to(device))

                    loss = args.lambda_adv_target[args.num_dataset - 1] * loss_adv_target \
                           + args.lambda_adv_memory[args.num_dataset - 1] * loss_adv_memory
                    loss_adv_target_value += loss_adv_target.item() / args.iter_size
                    loss_adv_memory_value += loss_adv_memory.item() / args.iter_size

                    loss = loss / args.iter_size
                    loss.backward()

                    # train D

                    # bring back requires_grad
                    for param in model_D.parameters():
                        param.requires_grad = True

                    # train with source
                    pred = pred_both.detach()

                    D_out = model_D(F.softmax(pred, dim=1))

                    loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(source_label).to(device))
                    loss_D = loss_D / args.iter_size / 2

                    loss_D.backward()

                    loss_D_value += loss_D.item()

                    # train with target
                    pred_target = pred_both_target.detach()

                    D_out = model_D(F.softmax(pred_target, dim=1))

                    loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(target_label).to(device))

                    loss_D = loss_D / args.iter_size / 2

                    loss_D.backward()
                    loss_D_value += loss_D.item()

                optimizer.step()
                optimizer_D.step()

                if args.tensorboard:
                    scalar_info = {
                        'loss_seg': loss_seg_value,
                        'loss_distillation': loss_distillation_value,
                        'loss_memory': loss_memory_value,
                        'loss_adv_target': loss_adv_target_value,
                        'loss_adv_memory': loss_adv_memory_value,
                        'loss_D': loss_D_value
                    }

                    if i_iter % 10 == 0:
                        for key, val in scalar_info.items():
                            writer.add_scalar(key, val, i_iter)

                print('exp = {}'.format(args.snapshot_dir))
                print(
                    'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_distill = {3:.3f} loss_memory = {4:.3f} loss_adv = {5:.3f} loss_adv_memory = {6:.3f} loss_D = {7:.3f}'.format(
                        i_iter, args.num_steps, loss_seg_value, loss_distillation_value, loss_memory_value, loss_adv_target_value, loss_adv_memory_value, loss_D_value))

                if args.num_dataset == 1:
                    if i_iter >= args.num_steps_stop - 1:
                        print('save model ...')
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_level_DM',
                                            'GTA5to' + str(args.target) + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_level_DM',
                                            'GTA5to' + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                        break

                    if i_iter % args.save_pred_every == 0 and i_iter != 0:
                        print('taking snapshot ...')
                        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'single_level_DM',
                                                                'GTA5to' + str(args.target) + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'single_level_DM',
                                                                  'GTA5to' + str(args.target) + str(i_iter) + '_D.pth'))
                else:
                    targetlist = list(dataset_dict.keys())
                    filename = 'GTA5to'
                    for i in range(args.num_dataset - 1):
                        filename += targetlist[i]
                        filename += 'to'
                    if i_iter >= args.num_steps_stop - 1:
                        print('save model ...')
                        torch.save(model.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_level_DM',
                                            filename + str(args.target) + str(args.num_steps_stop) + '.pth'))
                        torch.save(model_D.state_dict(),
                                   osp.join(args.snapshot_dir, 'single_level_DM',
                                            filename + str(args.target) + str(args.num_steps_stop) + '_D.pth'))
                        break

                    if i_iter % args.save_pred_every == 0 and i_iter != 0:
                        print('taking snapshot ...')
                        torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'single_level_DM',
                                                                filename + str(args.target) + str(i_iter) + '.pth'))
                        torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'single_level_DM',
                                                                  filename + str(args.target) + str(i_iter) + '_D.pth'))

            if args.tensorboard:
                writer.close()


if __name__ == '__main__':
    main()
