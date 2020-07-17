import numpy as np
import argparse
import json
import torch
import random
from PIL import Image
from os.path import join
from model.deeplab_DM import Deeplab_DM
from options import TrainOptions, dataset_list


SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping

RANDOM_SEED = 1338

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir=''):

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    PRE_TRAINED_SEG = './snapshots/OLD/Scratch_warper/single_level/GTA5_75000.pth'
    saved_state_dict = torch.load(PRE_TRAINED_SEG, map_location=device)

    opt = TrainOptions().parse()


    model = Deeplab_DM(args=opt)

    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        if i in new_params.keys():
            new_params[i] = saved_state_dict[i]
    model.load_state_dict(new_params)

    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    # for ind_class in range(num_classes):
    #     print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


def compute_iou(args):
    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        pred_dir = join(args.pred_dir, 'step' + str((files + 1) * args.save_pred_every))
        compute_mIoU(args.gt_dir, pred_dir, args.devkit_dir)

    # pred_dir = join(args.pred_dir, 'step' + str(115000))
    # compute_mIoU(args.gt_dir, pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # parser.add_argument('--gt_dir', type=str, default='/home/joonhkim/UDA/datasets/CityScapes/gtFine/val',
    #                     help='directory which stores CityScapes val gt images')
    parser.add_argument('--gt_dir', type=str, default='/work/CityScapes/gtFine/val',
                        help='directory which stores CityScapes val gt images')
    # parser.add_argument('--gt_dir', type=str, default='/home/smyoo/CAG_UDA/dataset/CityScapes/gtFine/val',
    #                     help='directory which stores CityScapes val gt images')
    # parser.add_argument('--gt_dir', type=str, default='/home/aiwc/Datasets/CityScapes/gtFine/val',
    #                     help='directory which stores CityScapes val gt images')

    parser.add_argument('--pred_dir', type=str, default='./result', help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")

    args = parser.parse_args()
    compute_iou(args)
