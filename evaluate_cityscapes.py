import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import random

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Deeplab
from model.deeplab_DM import Deeplab_DM
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn

SOURCE_ONLY = False
MEMORY = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 250000  # early stopping

dataset_dict = {'cityscapes': 1, 'synthia': 2}
TARGET = 'cityscapes'
NUM_DATASET = dataset_dict[TARGET]
INPUT_SIZE_TARGET = '1024,512'

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/work/CityScapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
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
                        help="available options : cityscapes, synthia")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--memory", action='store_true', default=MEMORY)
    parser.add_argument("--num-dataset", type=int, default=NUM_DATASET, help="Which target dataset?")
    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        if SOURCE_ONLY:
            model = Deeplab(num_classes=args.num_classes)
            saved_state_dict = torch.load('./snapshots/source_only/GTA5_' + str((files + 1) * args.save_pred_every) + '.pth')
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                # layer5.conv2d_list.3.weight
                i_parts = i.split('.')
                if not i_parts[0] == 'layer5':
                    new_params[i] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            if not args.memory:
                model = Deeplab(num_classes=args.num_classes)
                saved_state_dict = torch.load('./snapshots/single_level/GTA5tocityscapes' + str((files + 1) * args.save_pred_every) + '.pth')
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    # layer5.conv2d_list.3.weight
                    i_parts = i.split('.')
                    if not i_parts[0] == 'layer5':
                        new_params[i] = saved_state_dict[i]
                model.load_state_dict(new_params)
            else:
                model = Deeplab_DM(num_classes=args.num_classes, len_dataset=args.num_dataset, args=args)
                targetlist = list(dataset_dict.keys())
                filename = 'GTA5to'
                for i in range(args.num_dataset - 1):
                    filename += targetlist[i]
                    filename += 'to'
                saved_state_dict = torch.load('./snapshots/single_level_DM/' + filename + str(args.target) + str((files + 1) * args.save_pred_every) + '.pth')
                new_params = model.state_dict().copy()
                for i in saved_state_dict:
                    new_params[i] = saved_state_dict[i]
                model.load_state_dict(new_params)

        device = torch.device("cuda" if not args.cpu else "cpu")
        model = model.to(device)

        model.eval()
        if args.target == 'cityscapes':
            testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                            batch_size=1, shuffle=False, pin_memory=True)
        elif args.target == 'synthia':  # SYNTHIA dataloader 필요!!!
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

            if SOURCE_ONLY:
                output = model(image, input_size_target)
            else:
                if not args.memory:
                    output = model(image, input_size_target)
                else:
                    output, _, _ = model(image, input_size_target, args)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            output_col = colorize_mask(output)
            output = Image.fromarray(output)

            name = name[0].split('/')[-1]
            if SOURCE_ONLY:
                if not os.path.exists(os.path.join(args.save, 'source_only', 'step' + str((files + 1) * args.save_pred_every))):
                    os.makedirs(os.path.join(args.save, 'source_only', 'step' + str((files + 1) * args.save_pred_every)))
                output.save(os.path.join(args.save, 'source_only', 'step' + str((files + 1) * args.save_pred_every), name))
                output_col.save(os.path.join(args.save, 'source_only', 'step' + str((files + 1) * args.save_pred_every),
                                             name.split('.')[0] + '_color.png'))
            else:
                if not args.memory:
                    if not os.path.exists(
                            os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every))):
                        os.makedirs(
                            os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every)))
                    output.save(
                        os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every), name))
                    output_col.save(
                        os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every),
                                     name.split('.')[0] + '_color.png'))
                else:
                    targetlist = list(dataset_dict.keys())
                    foldername = 'GTA5to'
                    for i in range(args.num_dataset - 1):
                        foldername += targetlist[i]
                        foldername += 'to'
                    if not os.path.exists(
                            os.path.join(args.save, 'single_level_DM', foldername + str(args.target), 'step' + str((files + 1) * args.save_pred_every))):
                        os.makedirs(
                            os.path.join(args.save, 'single_level_DM', foldername + str(args.target), 'step' + str((files + 1) * args.save_pred_every)))
                    output.save(
                        os.path.join(args.save, 'single_level_DM', foldername + str(args.target), 'step' + str((files + 1) * args.save_pred_every), name))
                    output_col.save(
                        os.path.join(args.save, 'single_level_DM', foldername + str(args.target), 'step' + str((files + 1) * args.save_pred_every),
                                     name.split('.')[0] + '_color.png'))


if __name__ == '__main__':
    main()
