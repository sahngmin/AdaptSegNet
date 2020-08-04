import argparse

SOURCE_ONLY = False
MEMORY = False
WARPER = False
SPADE_WARPER = False
FROM_SCRATCH = True

SAVE_PRED_EVERY = 3000
NUM_STEPS_STOP = 30000 # early stopping
NUM_STEPS = 40000

dataset_list = ['GTA5', 'CityScapes']
TARGET = dataset_list[-1]
SET = 'train'
NUM_DATASET = len(dataset_list)

SAVE_DIR_NAME = 'single_alignment_hinge_cityscap_new_002_disc4'
SEGNET_NAME = 'checkpoint'
WARPER_NAME = 'SPADE'

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9

LEARNING_RATE_D = 1e-4

GAN = 'Hinge'

LAMBDA_ADV_TARGET = [0.002, 0.002]
LAMBDA_DISTILLATION = 0.1
UPDATE_DISC = 4

RANDOM_SEED = 1338

MODEL = 'DeepLab'
BATCH_SIZE = 5
ITER_SIZE = 1
NUM_WORKERS = 0
# DATA_DIRECTORY = '/home/joonhkim/UDA/datasets/GTA5'
# DATA_DIRECTORY = '/work/GTA5'
DATA_DIRECTORY = './dataset/GTA5'

# DATA_DIRECTORY = '/home/aiwc/Datasets/GTA5'
DATA_LIST_PATH = './data/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '512,256'
# DATA_DIRECTORY_TARGET = '/home/joonhkim/UDA/datasets/CityScapes'
# DATA_DIRECTORY_TARGET = '/work/CityScapes'
DATA_DIRECTORY_TARGET = './dataset/CityScapes'
DATA_LIST_PATH_TARGET = './data/cityscapes_list/train.txt'
INPUT_SIZE_TARGET = '512,256'

NUM_CLASSES = 13

RESTORE_FROM_RESNET = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
# RESTORE_FROM_RESNET = 'DeepLab_resnet_pretrained_init-f81d91e8.pth'

SAVE_NUM_IMAGES = 2

SNAPSHOT_DIR = './snapshots/'
LOG_DIR = './log'


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

        self.parser.add_argument("--model", type=str, default=MODEL,
                            help="available options : DeepLab")
        self.parser.add_argument("--target", type=str, default=TARGET,
                            help="available options : CityScapes, Synthia")
        self.parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        self.parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                            help="Accumulate gradients for ITER_SIZE iterations.")
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
        self.parser.add_argument("--lambda-adv-target", type=list, default=LAMBDA_ADV_TARGET,
                            help="lambda_adv for adversarial training.")
        self.parser.add_argument("--lambda-distillation", type=float, default=LAMBDA_DISTILLATION)
        self.parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Momentum component of the optimiser.")
        self.parser.add_argument("--not-restore-last", action="store_true",
                            help="Whether to not restore last (FC) layers.")
        self.parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                            help="Number of classes to predict (including background).")
        self.parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        self.parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                            help="Number of training steps for early stopping.")
        self.parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        self.parser.add_argument("--random-mirror", action="store_true",
                            help="Whether to randomly mirror the inputs during the training.")
        self.parser.add_argument("--random-scale", action="store_true",
                            help="Whether to randomly scale the inputs during the training.")
        self.parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")
        self.parser.add_argument("--restore-from-resnet", type=str, default=RESTORE_FROM_RESNET,
                            help="Where restore model parameters from.")
        self.parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                            help="How many images to save.")
        self.parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        self.parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        self.parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        self.parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
        self.parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.", default=True)
        self.parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                            help="Path to the directory of log.")
        self.parser.add_argument("--set", type=str, default=SET,
                            help="choose adaptation set.")
        self.parser.add_argument("--gan", type=str, default=GAN,
                            help="choose the GAN objective.")
        self.parser.add_argument("--source-only", action='store_true', default=SOURCE_ONLY)
        self.parser.add_argument("--memory", action='store_true', default=MEMORY)
        self.parser.add_argument("--from-scratch", action='store_true', default=FROM_SCRATCH)
        self.parser.add_argument("--num-dataset", type=int, default=NUM_DATASET, help="Which target dataset?")
        self.parser.add_argument("--input_size", default=INPUT_SIZE)
        self.parser.add_argument("--update_disc", default=UPDATE_DISC)


        self.parser.add_argument("--warper", action='store_true', default=WARPER)
        self.parser.add_argument("--feat-warp", default=True)
        self.parser.add_argument("--multi_gpu", default=False)
        self.parser.add_argument("--spadeWarper", default=SPADE_WARPER)

        self.parser.add_argument("--dir_name", default=SAVE_DIR_NAME)
        self.parser.add_argument("--segnet_name", default=SEGNET_NAME)
        self.parser.add_argument("--warper_name", default=WARPER_NAME)

        # SPADE options
        self.parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3',
                            help='instance normalization or batch normalization')
        self.parser.add_argument('--semantic_nc', type=int, default=NUM_CLASSES, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        self.parser.add_argument('--ngf', type=int, default=4, help='# of gen filters in first conv layer')
        self.parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")


