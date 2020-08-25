import argparse

FROM_SCRATCH = True
DM = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 150000  # early stopping
NUM_STEPS = 200000

SOURCE = 'GTA5'  # 'GTA5' or 'SYNTHIA'
TARGET = 'CityScapes'  # 'CityScapes' or 'IDD'
SET = 'train'

DIR_NAME = 'AdaptSegNet_new_Hinge(SpecO)_multi_DM'

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9

LEARNING_RATE_D = 1e-4

GAN = 'new_Hinge'  # 'Vanilla' or 'LS' or 'Hinge' or 'new_Hinge'
SPEC = True

LAMBDA_ADV2 = 0.001
LAMBDA_ADV1 = 0.0002
LAMBDA_SEG = 0.1
LAMBDA_DISTILL2 = 0.1
LAMBDA_DISTILL1 = 0.01

RANDOM_SEED = 1338

IGNORE_LABEL = 255

BATCH_SIZE = 1
NUM_WORKERS = 4

if SOURCE == 'GTA5':
    DATA_DIRECTORY = '/work/GTA5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    NUM_CLASSES = 18
elif SOURCE == 'SYNTHIA':
    DATA_DIRECTORY = '/work/SYNTHIA'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    NUM_CLASSES = 13
INPUT_SIZE = '1024,512'

if TARGET == 'CityScapes':
    DATA_DIRECTORY_TARGET = '/work/CityScapes'
    DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
    NUM_TARGET = 1
elif TARGET == 'IDD':
    DATA_DIRECTORY_TARGET = '/work/IDD_Segmentation'
    DATA_LIST_PATH_TARGET = './dataset/idd_list/train.txt'
    NUM_TARGET = 2
INPUT_SIZE_TARGET = '1024,512'
EVAL_TARGET = -1

RESTORE_FROM_RESNET = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

SNAPSHOT_DIR = './snapshots/'


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
        self.parser = parser

    def parse(self):
        # get the basic options
        opt = self.parser.parse_args()
        self.opt = opt

        return self.opt


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        # Training options

        self.parser.add_argument("--source", type=str, default=SOURCE)
        self.parser.add_argument("--target", type=str, default=TARGET)
        self.parser.add_argument("--num-target", type=int, default=NUM_TARGET)
        self.parser.add_argument("--eval-target", type=int, default=EVAL_TARGET)
        self.parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        self.parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                            help="number of workers for multithread dataloading.")
        self.parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                            help="Path to the directory containing the source dataset.")
        self.parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                            help="Path to the file listing the images in the source dataset.")
        self.parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")
        self.parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                            help="Comma-separated string with height and width of source images.")
        self.parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                            help="Path to the directory containing the target dataset.")
        self.parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                            help="Path to the file listing the images in the target dataset.")
        self.parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                            help="Comma-separated string with height and width of target images.")
        self.parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
        self.parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        self.parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                            help="Base learning rate for discriminator.")
        self.parser.add_argument("--lambda-adv1", type=float, default=LAMBDA_ADV1,
                            help="lambda_adv for adversarial training.")
        self.parser.add_argument("--lambda-adv2", type=float, default=LAMBDA_ADV2,
                                 help="lambda_adv for adversarial training.")
        self.parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                                 help="lambda_adv for adversarial training.")
        self.parser.add_argument("--lambda-distill1", type=float, default=LAMBDA_DISTILL1,
                                 help="lambda_distill for knowledge distillation.")
        self.parser.add_argument("--lambda-distill2", type=float, default=LAMBDA_DISTILL2,
                                 help="lambda_distill for knowledge distillation.")
        self.parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Whether to not restore last (FC) layers.")
        self.parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        self.parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        self.parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                            help="Number of training steps for early stopping.")
        self.parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        self.parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")
        self.parser.add_argument("--restore-from-resnet", type=str, default=RESTORE_FROM_RESNET,
                            help="Where restore model parameters from.")
        self.parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        self.parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        self.parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        self.parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
        self.parser.add_argument("--set", type=str, default=SET,
                            help="choose adaptation set.")
        self.parser.add_argument("--gan", type=str, default=GAN,
                            help="choose the GAN objective.")
        self.parser.add_argument("--spec", action='store_true', default=SPEC)
        self.parser.add_argument("--from-scratch", action='store_true', default=FROM_SCRATCH)
        self.parser.add_argument("--dm", action='store_true', default=DM)
        self.parser.add_argument("--dir-name", type=str, default=DIR_NAME)
