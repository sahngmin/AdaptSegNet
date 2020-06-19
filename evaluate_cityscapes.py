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
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import torch.nn as nn

SOURCE_ONLY = False
LEVEL = 'single-level'

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/aiwc/Datasets/CityScapes'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.

RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
# RESTORE_FROM = './snapshots/GTA5_50.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--level", type=str, default=LEVEL, help="single-level/multi-level")
    parser.add_argument("--multi-gpu", action='store_false')
    parser.add_argument("--warper", default=True)

    return parser.parse_args()


def main():
    seed = 1338
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    input_size = [2048, 1024]

    """Create the model and start the evaluation process."""

    args = get_arguments()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    # if args.restore_from[:4] == 'http' :
    #     saved_state_dict = model_zoo.load_url(args.restore_from)
    # else:
    #     saved_state_dict = torch.load(args.restore_from)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        if SOURCE_ONLY:
            saved_state_dict = torch.load('./snapshots/source_only/GTA5_' + str((files + 1) * args.save_pred_every) + '.pth')
        else:
            if args.level == 'single-level':
                saved_state_dict = torch.load('./snapshots/single_level/GTA5_' + str((files + 1) * args.save_pred_every) + '.pth')
            elif args.level == 'multi-level':
                saved_state_dict = torch.load('./snapshots/multi_level/GTA5_' + str((files + 1) * args.save_pred_every) + '.pth')
            else:
                raise NotImplementedError('level choice {} is not implemented'.format(args.level))
        ### for running different versions of pytorch
        model_dict = model.state_dict()
        saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        model_dict.update(saved_state_dict)
        ###
        model.load_state_dict(saved_state_dict)

        device = torch.device("cuda" if not args.cpu else "cpu")
        model = model.to(device)
        if args.multi_gpu:
            model = nn.DataParallel(model)

        model.eval()



        testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                        batch_size=1, shuffle=False, pin_memory=True)

        interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd' % index)
            image, _, name = batch
            image = image.to(device)

            if args.model == 'DeeplabMulti':
                output1, output2 = model(image, input_size, warping=args.warper)
                output = interp(output2).cpu().data[0].numpy()
            elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
                output = model(image)
                output = interp(output).cpu().data[0].numpy()

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
                if args.level == 'single-level':
                    if not os.path.exists(
                            os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every))):
                        os.makedirs(
                            os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every)))
                    output.save(
                        os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every), name))
                    output_col.save(
                        os.path.join(args.save, 'single_level', 'step' + str((files + 1) * args.save_pred_every),
                                     name.split('.')[0] + '_color.png'))
                elif args.level == 'multi-level':
                    if not os.path.exists(
                            os.path.join(args.save, 'multi_level', 'step' + str((files + 1) * args.save_pred_every))):
                        os.makedirs(
                            os.path.join(args.save, 'multi_level', 'step' + str((files + 1) * args.save_pred_every)))
                    output.save(
                        os.path.join(args.save, 'multi_level', 'step' + str((files + 1) * args.save_pred_every), name))
                    output_col.save(
                        os.path.join(args.save, 'multi_level', 'step' + str((files + 1) * args.save_pred_every),
                                     name.split('.')[0] + '_color.png'))
                else:
                    raise NotImplementedError('level choice {} is not implemented'.format(args.level))



if __name__ == '__main__':
    main()
