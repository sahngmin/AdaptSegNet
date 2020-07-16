import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_DM import Deeplab_DM
from dataset.synthia_dataset import SYNTHIADataSet
from collections import OrderedDict
import os
import os.path as osp
from PIL import Image

SOURCE_ONLY = False
MEMORY = True
WARPER = False

TRAIN_VAL = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping

IMG_MEAN = np.array((0, 0, 0), dtype=np.float32)

# DATA_DIRECTORY = '/home/joonhkim/UDA/datasets/SYNTHIA-SEQS-04-SPRING'
DATA_DIRECTORY = '/work/SYNTHIA-SEQS-04-SPRING'
DATA_LIST_PATH = './dataset/synthia_seqs_04_spring_list/val.txt'

# DATA_DIRECTORY_TARGET = '/home/joonhkim/UDA/datasets/SYNTHIA-SEQS-02-SPRING'
DATA_DIRECTORY_TARGET = '/work/SYNTHIA-SEQS-02-SPRING'
DATA_LIST_PATH_TARGET = './dataset/synthia_seqs_02_spring_list/val.txt'

dataset_dict = {'SEQS-04-SPRING': 0, 'SEQS-02-SPRING': 1, 'SEQS-01-SPRING': 2}
SOURCE = 'SEQS-04-SPRING'
TARGET = 'SEQS-02-SPRING'
NUM_DATASET = dataset_dict[TARGET]

IGNORE_LABEL = 255
NUM_CLASSES = 11
BATCH_SIZE = 1

SET = 'val'

RANDOM_SEED = 1338

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")

    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--memory", action='store_true', default=MEMORY)
    parser.add_argument("--num-dataset", type=int, default=NUM_DATASET, help="Which target dataset?")
    parser.add_argument("--source-only", action='store_true', default=SOURCE_ONLY)
    parser.add_argument("--warper", action='store_true', default=WARPER)
    parser.add_argument("--feat-warp", default=True)


    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (1024, 512)

    name_classes = np.asarray(['Sky', 'Building', 'Road', 'Sidewalk', 'Fence', 'Vegetation', 'Pole', 'Car',
                               'Traffic Sign', 'Lanemarking', 'Traffic Light'])

    """Create the model and start the evaluation process."""

    args = get_arguments()

    model = Deeplab_DM(args=args)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)

        if args.source_only:
            if args.warper and args.memory:
                saved_state_dict = torch.load('./snapshots/source_only_warp_DM/' + str(args.source) + '_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
            elif args.warper:
                saved_state_dict = torch.load('./snapshots/source_only_warp/' + str(args.source) + '_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
            elif args.memory:
                saved_state_dict = torch.load('./snapshots/source_only_DM/' + str(args.source) + '_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
                # saved_state_dict = torch.load('./snapshots/source_only_DM/' + str(args.source) + '_'
                #                               + str(150000) + '.pth')
            else:
                saved_state_dict = torch.load('./snapshots/source_only/' + str(args.source) + '_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
        else:
            if args.num_dataset == 1:
                if args.warper and args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp_DM/' + str(args.source) + 'to'
                                                  + str(args.target) + '_'
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.warper:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp/' + str(args.source) + 'to'
                                                  + str(args.target) + '_'
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_DM/' + str(args.source) + 'to'
                                                  + str(args.target) + '_'
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                    # saved_state_dict = torch.load('./snapshots/single_alignment_DM/' + str(args.source) + 'to'
                    #                               + str(args.target) + '_'
                    #                               + str(150000) + '.pth')
                else:
                    saved_state_dict = torch.load('./snapshots/single_alignment/' + str(args.source) + 'to'
                                                  + str(args.target) + '_'
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
            else:
                targetlist = list(dataset_dict.keys())
                filename = str(args.source) + 'to'
                for i in range(args.num_dataset - 1):
                    filename += targetlist[i]
                    filename += 'to'
                if args.warper and args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp_DM/' + filename + str(args.target)
                                                  + '_' + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.warper:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp/' + filename + str(args.target)
                                                  + '_' + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_DM/' + filename + str(args.target)
                                                  + '_' + str((files + 1) * args.save_pred_every) + '.pth')
                else:
                    saved_state_dict = torch.load('./snapshots/single_alignment/' + filename + str(args.target)
                                                  + '_' + str((files + 1) * args.save_pred_every) + '.pth')

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()
        if TRAIN_VAL:
            train_val_loader = torch.utils.data.DataLoader(
                SYNTHIADataSet(args.data_dir, args.data_list,
                               max_iters=None,
                               crop_size=input_size,
                               scale=False, mirror=False,
                               mean=IMG_MEAN, set=args.set),
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
            # for ind_class in range(args.num_classes):
            #     print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Source): ' + str(round(np.nanmean(mIoUs) * 100, 2)))

        val_val_loader = torch.utils.data.DataLoader(
            SYNTHIADataSet(args.data_dir_target, args.data_list_target,
                           max_iters=None,
                           crop_size=input_size,
                           scale=False, mirror=False,
                           mean=IMG_MEAN, set=args.set),
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
        # for ind_class in range(args.num_classes):
        #     print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===> mIoU (Target): ' + str(round(np.nanmean(mIoUs) * 100, 2)))


if __name__ == '__main__':
    main()
