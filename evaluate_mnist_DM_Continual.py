import argparse
import numpy as np
import random
import pdb
import torch
from torch.utils import data
from model.MNIST_model import AlexNet_DM, AlexNet_Source
from data.DigitFive.datasets.dataset_read import dataset_read


DIR_NAME = 'mnist_'
TARGET = 'usps'

MNIST = True
USPS = True
SYN = True
MNISTM = True
SVHN = True

LAMBDA_ADV = 1.0


SAVE_PRED_EVERY = 600
NUM_STEPS_STOP = 6000

BATCH_SIZE = 16

IGNORE_LABEL = 255

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
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--mnist", action='store_true', default=MNIST)
    parser.add_argument("--usps", action='store_true', default=USPS)
    parser.add_argument("--syn", action='store_true', default=SYN)
    parser.add_argument("--mnistm", action='store_true', default=MNISTM)
    parser.add_argument("--svhn", action='store_true', default=SVHN)

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
    parser.add_argument("--dir-name", type=str, default=DIR_NAME)
    parser.add_argument("--case", type=int, default=4)
    parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                             help="lambda_adv for adversarial training.")
    parser.add_argument("--gan", type=str, default='Hinge')

    parser.add_argument("--continual", action='store_true', default=True)

    return parser.parse_args()


def main():
    args = get_arguments()

    continual_list = ['mnist', 'usps', 'mnistm', 'syn', 'svhn']

    if args.case == 1:
        args.target = continual_list[1]
        num_target = 1
        if args.continual:
            args.dir_name = args.dir_name + args.target + '1.0' + args.gan

    elif args.case == 2:
        args.target = continual_list[2]
        num_target = 2
        if args.continual:
            args.dir_name = args.dir_name + args.target + '3.0' + args.gan

    elif args.case == 3:
        args.target = continual_list[3]
        num_target = 3
        if args.continual:
            args.dir_name = args.dir_name + args.target + str(args.lambda_adv) + args.gan

    elif args.case == 4:
        args.target = continual_list[4]
        num_target = 4
        if args.continual:
            args.dir_name = args.dir_name + args.target + '3.0' + args.gan

    else:
        num_target = continual_list.index(args.target)

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 멀티 gpu 연산 무작위 고정
    torch.backends.cudnn.enabled = False  # cudnn library를 사용하지 않게 만듬
    np.random.seed(seed)
    random.seed(seed)

    # Create the model and start the evaluation process
    model = AlexNet_DM(num_target=num_target)

    if not args.continual:
        args.dir_name = args.dir_name + args.target + str(args.lambda_adv) + args.gan


    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print(args.dir_name)
        print('Step: ', (files + 1) * args.save_pred_every)
        saved_state_dict = torch.load('./snapshots/' + args.dir_name + '/' + str((files + 1) * args.save_pred_every) + '.pth')
        # saved_state_dict = torch.load('./snapshots/' + '30000.pth')
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            if i in new_params.keys():
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()

        if args.mnist:
            targetloader, test_dataloader = dataset_read('mnist', args.batch_size)
            count = 0
            correct = 0
            for i, data in enumerate(test_dataloader):
                images_val, labels = data
                images_val, labels = images_val.to(device), labels.to(device)
                feat_new, feat_ori, pred, output_ori = model(images_val)
                _, pred = pred.max(dim=1)
                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()
                correct += (pred == labels).sum()
                count += pred.shape[0]

            acc_val_1 = correct / count
            if not args.continual:
                print('MNIST')
            print(str(round(np.nanmean(acc_val_1) * 100, 2)))
        if args.usps:
            targetloader, test_dataloader = dataset_read('usps', args.batch_size)
            count = 0
            correct = 0
            for i, data in enumerate(test_dataloader):
                images_val, labels = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.case > 1:
                    num_idx = 1
                else:
                    num_idx = None
                feat_new, feat_ori, pred, output_ori = model(images_val, dm_idx=num_idx)
                _, pred = pred.max(dim=1)
                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()
                correct += (pred == labels).sum()
                count += pred.shape[0]

            acc_val_1 = correct / count
            if not args.continual:
                print('USPS')
            print(str(round(np.nanmean(acc_val_1) * 100, 2)))

        if args.mnistm:
            targetloader, test_dataloader = dataset_read('mnistm', args.batch_size)
            count = 0
            correct = 0
            for i, data in enumerate(test_dataloader):
                images_val, labels = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.case > 2:
                    num_idx = 2
                else:
                    num_idx = None
                feat_new, feat_ori, pred, output_ori = model(images_val, dm_idx=num_idx)
                _, pred = pred.max(dim=1)
                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()
                correct += (pred == labels).sum()
                count += pred.shape[0]

            acc_val_1 = correct / count
            if not args.continual:
                print('MNISTM')
            print(str(round(np.nanmean(acc_val_1) * 100, 2)))


        if args.syn:
            targetloader, test_dataloader = dataset_read('syn', args.batch_size)
            count = 0
            correct = 0
            for i, data in enumerate(test_dataloader):
                images_val, labels = data
                images_val, labels = images_val.to(device), labels.to(device)
                if args.case > 3:
                    num_idx = 3
                else:
                    num_idx = None
                feat_new, feat_ori, pred, output_ori = model(images_val, dm_idx=num_idx)
                _, pred = pred.max(dim=1)
                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()
                correct += (pred == labels).sum()
                count += pred.shape[0]
            if not args.continual:
                print('SYN')
            acc_val_1 = correct / count
            print(str(round(np.nanmean(acc_val_1) * 100, 2)))

        if args.svhn:
            targetloader, test_dataloader = dataset_read('svhn', args.batch_size)
            count = 0
            correct = 0
            for i, data in enumerate(test_dataloader):
                images_val, labels = data
                images_val, labels = images_val.to(device), labels.to(device)
                feat_new, feat_ori, pred, output_ori = model(images_val)
                _, pred = pred.max(dim=1)
                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()
                correct += (pred == labels).sum()
                count += pred.shape[0]

            acc_val_1 = correct / count
            if not args.continual:
                print('SVHN')
            print(str(round(np.nanmean(acc_val_1) * 100, 2)))

        print('\n')


if __name__ == '__main__':
    main()
