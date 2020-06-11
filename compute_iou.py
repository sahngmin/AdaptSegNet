import numpy as np
import argparse
import json
import torch
import random
from PIL import Image
from os.path import join

SOURCE_ONLY = True
LEVEL = 'single-level'

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping

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
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    # print('Num classes', num_classes)
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


def main(args):
    seed = 1338
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        if SOURCE_ONLY:
            pred_dir = join(args.pred_dir, 'source_only', 'step' + str((files + 1) * args.save_pred_every))
        else:
            if args.level == 'single-level':
                pred_dir = join(args.pred_dir, 'single_level', 'step' + str((files + 1) * args.save_pred_every))
            elif args.level == 'multi-level':
                pred_dir = join(args.pred_dir, 'multi-level', 'step' + str((files + 1) * args.save_pred_every))
            else:
                raise NotImplementedError('level choice {} is not implemented'.format(args.level))
        compute_mIoU(args.gt_dir, pred_dir, args.devkit_dir)
    # compute_mIoU(args.gt_dir, args.pred_dir, args.devkit_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, default='/work/CityScapes/gtFine/val', help='directory which stores CityScapes val gt images')
    parser.add_argument('--pred_dir', type=str, default='./result/cityscapes', help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='dataset/cityscapes_list', help='base directory of cityscapes')

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")

    parser.add_argument("--level", type=str, default=LEVEL, help="single-level/multi-level")

    args = parser.parse_args()
    main(args)
