import argparse
import numpy as np
import random

import torch
from torch.utils import data
from model.deeplab_DM import Deeplab_DM
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet

SOURCE = 'GTA5'  # 'GTA5' or 'SYNTHIA'
NUM_TARGET = 1

GTA5 = True
SYNTHIA = True
CityScapes = True
IDD = True
PER_CLASS = True

SAVE_PRED_EVERY = 3000
NUM_STEPS_STOP = 30000

BATCH_SIZE = 6

DATA_DIRECTORY_GTA5 = '/work/GTA5'
DATA_LIST_PATH_GTA5 = './dataset/gta5_list/val.txt'

DATA_DIRECTORY_SYNTHIA = '/work/SYNTHIA'
DATA_LIST_PATH_SYNTHIA = './dataset/synthia_list/val.txt'

DATA_DIRECTORY_CityScapes = '/work/CityScapes'
DATA_LIST_PATH_CityScapes = './dataset/cityscapes_list/val.txt'

DATA_DIRECTORY_IDD = '/work/IDD_Segmentation'
DATA_LIST_PATH_IDD = './dataset/idd_list/val.txt'

IGNORE_LABEL = 255

if SOURCE == 'GTA5':
    NUM_CLASSES = 18
elif SOURCE == 'SYNTHIA':
    NUM_CLASSES = 13

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
    parser.add_argument("--source", action='store_true', default=SOURCE)
    parser.add_argument("--num-target", type=int, default=NUM_TARGET)
    parser.add_argument("--gta5", action='store_true', default=GTA5)
    parser.add_argument("--synthia", action='store_true', default=SYNTHIA)
    parser.add_argument("--cityscapes", action='store_true', default=CityScapes)
    parser.add_argument("--idd", action='store_true', default=IDD)
    parser.add_argument("--mIoUs-per-class", action='store_true', default=PER_CLASS)
    parser.add_argument("--data-dir-gta5", type=str, default=DATA_DIRECTORY_GTA5)
    parser.add_argument("--data-list-gta5", type=str, default=DATA_LIST_PATH_GTA5)
    parser.add_argument("--data-dir-synthia", type=str, default=DATA_DIRECTORY_SYNTHIA)
    parser.add_argument("--data-list-synthia", type=str, default=DATA_LIST_PATH_SYNTHIA)
    parser.add_argument("--data-dir-cityscapes", type=str, default=DATA_DIRECTORY_CityScapes)
    parser.add_argument("--data-list-cityscapes", type=str, default=DATA_LIST_PATH_CityScapes)
    parser.add_argument("--data-dir-idd", type=str, default=DATA_DIRECTORY_IDD)
    parser.add_argument("--data-list-idd", type=str, default=DATA_LIST_PATH_IDD)
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

    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 gpu 연산 무작위 고정
    torch.backends.cudnn.enabled = False  # cudnn library를 사용하지 않게 만듬
    np.random.seed(seed)
    random.seed(seed)

    input_size = (512, 256)

    if args.num_classes == 13:
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
    elif args.num_classes == 18:
        name_classes = np.asarray(["road",
                                   "sidewalk",
                                   "building",
                                   "wall",
                                   "fence",
                                   "pole",
                                   "light",
                                   "sign",
                                   "vegetation",
                                   "sky",
                                   "person",
                                   "rider",
                                   "car",
                                   "truck",
                                   "bus",
                                   "train",
                                   "motorcycle",
                                   "bicycle"])
    else:
        NotImplementedError("Unavailable number of classes")

    # Create the model and start the evaluation process
    model = Deeplab_DM(args=args)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        saved_state_dict = torch.load('./snapshots/' + str((files + 1) * args.save_pred_every) + '.pth')
        # saved_state_dict = torch.load('./snapshots/' + '30000.pth')
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()
        if args.gta5:
            gta5_loader = torch.utils.data.DataLoader(
                GTA5DataSet(args.data_dir_gta5, args.data_list_gta5,
                            crop_size=input_size, ignore_label=args.ignore_label,
                            set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(gta5_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred, _ = model(images_val, input_size)
                # pred = nn.Upsample(size=(1052, 1914), mode='bilinear', align_corners=True)(pred)
                # labels = labels.unsqueeze(1)
                # labels = nn.Upsample(size=(1052, 1914), mode='nearest')(labels)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (GTA5): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.synthia:
            synthia_loader = torch.utils.data.DataLoader(
                SYNTHIADataSet(args.data_dir_synthia, args.data_list_synthia,
                               crop_size=input_size, ignore_label=args.ignore_label,
                               set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(synthia_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred, _ = model(images_val, input_size)
                # pred = nn.Upsample(size=(760, 1280), mode='bilinear', align_corners=True)(pred)
                # labels = labels.unsqueeze(1)
                # labels = nn.Upsample(size=(760, 1280), mode='nearest')(labels)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (SYNTHIA): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.cityscapes:
            cityscapes_loader = torch.utils.data.DataLoader(
                cityscapesDataSet(args.data_dir_cityscapes, args.data_list_cityscapes,
                                  crop_size=input_size, ignore_label=args.ignore_label,
                                  set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(cityscapes_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred, _ = model(images_val, input_size)
                # pred = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)(pred)
                # labels = labels.unsqueeze(1)
                # labels = nn.Upsample(size=(1024, 2048), mode='nearest')(labels)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (CityScapes): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.idd:
            idd_loader = torch.utils.data.DataLoader(
                IDDDataSet(args.data_dir_idd, args.data_list_idd,
                           crop_size=input_size, ignore_label=args.ignore_label,
                           set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(idd_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred, _ = model(images_val, input_size)
                # pred = nn.Upsample(size=(1080, 1920), mode='bilinear', align_corners=True)(pred)
                # labels = labels.unsqueeze(1)
                # labels = nn.Upsample(size=(1080, 1920), mode='nearest')(labels)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (IDD): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

if __name__ == '__main__':
    main()
