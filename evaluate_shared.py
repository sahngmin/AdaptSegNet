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
from model.deeplab import Deeplab
from model.deeplab_DM import Deeplab_DM
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SYNTHIADataSet
from data.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
import os.path as osp
from PIL import Image

SOURCE_ONLY = False
MEMORY = False
WARPER = False
NUM_DATASET = 1

DATA_DIRECTORY = './dataset/GTA5'
DATA_LIST_PATH = './data/gta5_list/val.txt'

DATA_DIRECTORY_TARGET1 = '/dataset/SYNTHIA'
DATA_LIST_PATH_TARGET1 = './data/synthia_list/val.txt'

DATA_DIRECTORY_TARGET2 = '/dataset/CityScapes'
DATA_LIST_PATH_TARGET2 = './data/cityscapes_list/val.txt'

IGNORE_LABEL = 255
NUM_CLASSES = 13
BATCH_SIZE = 1
INPUT_SIZE = '512,256'
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

    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--data-dir-target1", type=str, default=DATA_DIRECTORY_TARGET1,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target1", type=str, default=DATA_LIST_PATH_TARGET1,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-dir-target2", type=str, default=DATA_DIRECTORY_TARGET2,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target2", type=str, default=DATA_LIST_PATH_TARGET2,
                        help="Path to the file listing the images in the target dataset.")

    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")

    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--memory", action='store_true', default=MEMORY)
    parser.add_argument("--num-dataset", type=int, default=NUM_DATASET, help="Which target dataset?")
    parser.add_argument("--source-only", action='store_true', default=SOURCE_ONLY)
    parser.add_argument("--warper", action='store_true', default=WARPER)
    parser.add_argument("--feat-warp", default=True)
    parser.add_argument("--multi_gpu", default=True)
    parser.add_argument("--input_size", default=INPUT_SIZE)
    parser.add_argument("--spadeWarper", default=False)


    return parser.parse_args()


def main():

    checkpoint_path = "./snapshots/single_alignment_hinge"
    args = get_arguments()
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (512, 256)

    name_classes = np.asarray(["road",
                               "sidewalk",
                               "building",
                               "light",
                               "sign",
                               "vegetation",
                               "sky",
                               "person",
                               "rider",
                               "car",
                               "bus",
                               "motocycle",
                               "bicycle"])

    model = Deeplab_DM(args=args)

    for model_name in os.listdir(checkpoint_path):
        checkpoint = torch.load(os.path.join(checkpoint_path, model_name))

        model.load_state_dict(checkpoint['state_dict_seg'])

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()

        """ Source (GTA5) """
        source_loader = torch.utils.data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list,
                        crop_size=input_size,
                        ignore_label=args.ignore_label, set=args.set),
            batch_size=args.batch_size, shuffle=False, pin_memory=True)

        hist = np.zeros((args.num_classes, args.num_classes))
        for i, data in enumerate(source_loader):
            images_val, labels, _ = data
            images_val, labels = images_val.to(device), labels.to(device)
            if args.warper and args.memory:
                pred_warped, _, pred, pred_ori = model(images_val, input_size)
            elif args.warper:
                _, pred_warped, _, pred = model(images_val, input_size)
            elif args.memory:
                _, _, pred, pred_ori = model(images_val, input_size)
            else:
                _, _, _, pred = model(images_val, input_size)

            pred = nn.Upsample(size=(1052, 1914), mode='bilinear', align_corners=True)(pred)
            labels = labels.unsqueeze(1)
            labels = nn.Upsample(size=(1052, 1914), mode='nearest')(labels)
            _, pred = pred.max(dim=1)

            labels = labels.cpu().numpy()
            pred = pred.cpu().detach().numpy()

            hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
        mIoUs = per_class_iu(hist)
        for ind_class in range(args.num_classes):
            print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===> mIoU (Source): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        print('=' * 50)

        """ Target1 (SYNTHIA) """
        target1_loader = torch.utils.data.DataLoader(
            SYNTHIADataSet(args.data_dir_target1, args.data_list_target1,
                           crop_size=input_size,
                           ignore_label=args.ignore_label, set=args.set),
            batch_size=args.batch_size, shuffle=False, pin_memory=True)

        hist = np.zeros((args.num_classes, args.num_classes))
        for i, data in enumerate(target1_loader):
            images_val, labels, _ = data
            images_val, labels = images_val.to(device), labels.to(device)
            if args.warper and args.memory:
                pred_warped, _, pred, pred_ori = model(images_val, input_size)
            elif args.warper:
                _, pred_warped, _, pred = model(images_val, input_size)
            elif args.memory:
                _, _, pred, pred_ori = model(images_val, input_size)
            else:
                _, _, _, pred = model(images_val, input_size)

            pred = nn.Upsample(size=(760, 1280), mode='bilinear', align_corners=True)(pred)
            labels = labels.unsqueeze(1)
            labels = nn.Upsample(size=(760, 1280), mode='nearest')(labels)
            _, pred = pred.max(dim=1)

            labels = labels.cpu().numpy()
            pred = pred.cpu().detach().numpy()

            hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
        mIoUs = per_class_iu(hist)
        for ind_class in range(args.num_classes):
            print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===> mIoU (Target1): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        print('=' * 50)

        """ Target2 (CityScapes) """
        target2_loader = torch.utils.data.DataLoader(
            cityscapesDataSet(args.data_dir_target2, args.data_list_target2,
                              crop_size=input_size,
                              ignore_label=args.ignore_label, set=args.set),
            batch_size=args.batch_size, shuffle=False, pin_memory=True)

        hist = np.zeros((args.num_classes, args.num_classes))
        for i, data in enumerate(target2_loader):
            images_val, labels, _ = data
            images_val, labels = images_val.to(device), labels.to(device)
            if args.warper and args.memory:
                pred_warped, _, pred, pred_ori = model(images_val, input_size)
            elif args.warper:
                _, pred_warped, _, pred = model(images_val, input_size)
            elif args.memory:
                _, _, pred, pred_ori = model(images_val, input_size)
            else:
                _, _, _, pred = model(images_val, input_size)

            pred = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)(pred)
            labels = labels.unsqueeze(1)
            labels = nn.Upsample(size=(1024, 2048), mode='nearest')(labels)
            _, pred = pred.max(dim=1)

            labels = labels.cpu().numpy()
            pred = pred.cpu().detach().numpy()

            hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
        mIoUs = per_class_iu(hist)
        for ind_class in range(args.num_classes):
            print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
        print('===> mIoU (Target2): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
        print('=' * 50)

if __name__ == '__main__':
    main()