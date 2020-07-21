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
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
import os.path as osp
from PIL import Image
from compute_iou import compute_iou

SOURCE_ONLY = False
MEMORY = True
WARPER = False

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# DATA_DIRECTORY = '/home/joonhkim/UDA/datasets/CityScapes'
# DATA_DIRECTORY = '/work/CityScapes'
DATA_DIRECTORY = '/home/smyoo/CAG_UDA/dataset/CityScapes'
# DATA_DIRECTORY = '/home/aiwc/Datasets/CityScapes'

GT_DIR = '/home/smyoo/CAG_UDA/dataset/CityScapes/gtFine/val'

dataset_dict = {'GTA5': 0, 'CityScapes': 1, 'Synthia': 2}
TARGET = 'CityScapes'
NUM_DATASET = dataset_dict[TARGET]

DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result'

IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 1
NUM_STEPS = 500 # Number of images in the validation set.

SET = 'val'

RANDOM_SEED = 1338

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : Cityscapes, Synthia")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                             help="Number of images sent to the network in one step.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
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

    parser.add_argument('--gt_dir', type=str, default=GT_DIR)
    # parser.add_argument('--gt_dir', type=str, default='/home/smyoo/CAG_UDA/dataset/CityScapes/gtFine/val',
    #                     help='directory which stores CityScapes val gt images')
    # parser.add_argument('--gt_dir', type=str, default='/home/aiwc/Datasets/CityScapes/gtFine/val',
    #                     help='directory which stores CityScapes val gt images')

    parser.add_argument('--pred_dir', type=str, default='./result',
                        help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')


    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = [512, 256]

    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = Deeplab_DM(args=args)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)

        if args.source_only:
            if args.warper and args.memory:
                saved_state_dict = torch.load('./snapshots/source_only_warp_DM/GTA5_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
            elif args.warper:
                saved_state_dict = torch.load('./snapshots/source_only_warp/GTA5_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
            elif args.memory:
                saved_state_dict = torch.load('./snapshots/source_only_DM/GTA5_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
            else:
                saved_state_dict = torch.load('./snapshots/source_only/GTA5_'
                                              + str((files + 1) * args.save_pred_every) + '.pth')
        else:
            if args.num_dataset == 1:
                if args.warper and args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp_DM/' + 'GTA5to' + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.warper:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp/' + 'GTA5to' + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_DM/' + 'GTA5to' + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                else:
                    saved_state_dict = torch.load('./snapshots/single_alignment/' + 'GTA5to' + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
            else:
                targetlist = list(dataset_dict.keys())
                filename = 'GTA5to'
                for i in range(args.num_dataset - 1):
                    filename += targetlist[i]
                    filename += 'to'
                if args.warper and args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp_DM/' + filename + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.warper:
                    saved_state_dict = torch.load('./snapshots/single_alignment_warp/' + filename + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                elif args.memory:
                    saved_state_dict = torch.load('./snapshots/single_alignment_DM/' + filename + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')
                else:
                    saved_state_dict = torch.load('./snapshots/single_alignment/' + filename + str(args.target)
                                                  + str((files + 1) * args.save_pred_every) + '.pth')

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()
        if args.target == 'CityScapes':
            testloader = data.DataLoader(
                cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                                  mirror=False, set=args.set),
                                  batch_size=1, shuffle=False, pin_memory=True)
        elif args.target == 'Synthia': #SYNTHIA dataloader 필요!!!---------------------------------------------------------------------------------------------
            testloader = data.DataLoader(
                cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False,
                                  mirror=False, set=args.set),
                                  batch_size=1, shuffle=False, pin_memory=True)
        else:
            raise NotImplementedError('Unavailable target domain')


        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd' % index)
            image, _, name = batch
            image = image.to(device)

            if args.warper and args.memory:
                output, _, _, _ = model(image, input_size)
            elif args.warper:
                _, output, _, _ = model(image, input_size)
            elif args.memory:
                _, _, output, _ = model(image, input_size)
            else:
                _, _, _, output = model(image, input_size)
            output = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)(output)

            output = output.cpu().data[0].numpy()

            output = output.transpose(1, 2, 0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]

            if not os.path.exists(os.path.join(args.save, 'step' + str((files + 1) * args.save_pred_every))):
                os.makedirs(os.path.join(args.save, 'step' + str((files + 1) * args.save_pred_every)))
            output.save(os.path.join(args.save, 'step' + str((files + 1) * args.save_pred_every), name))
            output_col.save(os.path.join(args.save, 'step' + str((files + 1) * args.save_pred_every),
                                         name.split('.')[0] + '_color.png'))

    compute_iou(args)

if __name__ == '__main__':
    main()
