import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils import data
from model.deeplab import Deeplab
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.synthia_dataset import SYNTHIADataSet

DATA_DIRECTORY = '/home/joonhkim/UDA/datasets/GTA5'
DATA_LIST_PATH = '/home/joonhkim/UDA/AdaptSegNet/dataset/gta5_list/val.txt'

DATA_DIRECTORY_TARGET = '/home/joonhkim/UDA/datasets/SYNTHIA'
DATA_LIST_PATH_TARGET = '/home/joonhkim/UDA/AdaptSegNet/dataset/synthia_list/val.txt'

TARGET2 = False
DATA_DIRECTORY_TARGET2 = '/home/joonhkim/UDA/datasets/CityScapes'
DATA_LIST_PATH_TARGET2 = 'home/joonhkim/UDA/AdaptSegNet/dataset/cityscapes_list/val.txt'

FILENAME = './snapshots/GTA5toSYNTHIA_50000.pth'
# FILENAME = './snapshots/GTA5toSYNTHIAtoCityScapes_50000.pth'

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_arguments():
    parser = argparse.ArgumentParser(description="ACE evaluation")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                             help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                             help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--data-dir-target2", type=str, default=DATA_DIRECTORY_TARGET2,
                             help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target2", type=str, default=DATA_LIST_PATH_TARGET2,
                             help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--num-classes", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--set", type=str, default='val')

    parser.add_argument("--random-seed", type=int, default=1338)

    return parser.parse_args()


def main():
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
                               "motorcycle",
                               "bicycle"])

    # Create the model and start the evaluation process

    model = Deeplab(args=args)

    saved_state_dict = torch.load(FILENAME)
    model.load_state_dict(saved_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()

    source_loader = torch.utils.data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list,
                    crop_size=input_size,
                    ignore_label=args.ignore_label, set=args.set),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    hist = np.zeros((args.num_classes, args.num_classes))
    for i, data in enumerate(source_loader):
        images_val, labels, _ = data
        images_val, labels = images_val.to(device), labels.to(device)
        pred = model(images_val, input_size)
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


    target1_loader = torch.utils.data.DataLoader(
        SYNTHIADataSet(args.data_dir_target, args.data_list_target,
                       crop_size=input_size,
                       ignore_label=args.ignore_label, set=args.set),
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

    hist = np.zeros((args.num_classes, args.num_classes))
    for i, data in enumerate(target1_loader):
        images_val, labels, _ = data
        images_val, labels = images_val.to(device), labels.to(device)
        pred = model(images_val, input_size)
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

    if TARGET2:
        target2_loader = torch.utils.data.DataLoader(
            cityscapesDataSet(args.data_dir_target2, args.data_list_target2,
                              crop_size=input_size,
                              ignore_label=args.ignore_label, set=args.set),
            batch_size=args.batch_size, shuffle=False, pin_memory=True)

        hist = np.zeros((args.num_classes, args.num_classes))
        for i, data in enumerate(target2_loader):
            images_val, labels, _ = data
            images_val, labels = images_val.to(device), labels.to(device)
            pred = model(images_val, input_size)
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
