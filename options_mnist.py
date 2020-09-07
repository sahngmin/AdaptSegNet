import argparse

SOURCE_ONLY = False
FROM_SCRATCH = True

SAVE_PRED_EVERY = 600
NUM_STEPS_STOP = 6000 # early stopping
NUM_STEPS = 10000

SET = 'train'

DIR_NAME = 'mnist_'
TARGET = 'usps'
PRE_TRAINED = ''

NUM_TARGET = 1

# LEARNING_RATE = 1e-3
# LEARNING_RATE_D = 5e-4

LEARNING_RATE = 1e-3
LEARNING_RATE_D = 1e-3

MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
POWER = 0.9

GAN = 'Hinge'  # 'Vanilla' or 'LS' or 'Hinge'

LAMBDA_ADV = 4.0
LAMBDA_DISTILL = 1.0

RANDOM_SEED = 1338

IGNORE_LABEL = 255

BATCH_SIZE = 8
NUM_WORKERS = 2

LOG_DIR = 'log'


SNAPSHOT_DIR = './snapshots'


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

        self.parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                            help="Number of images sent to the network in one step.")
        self.parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                            help="number of workers for multithread dataloading.")

        self.parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                            help="The index of the label to ignore during the training.")

        self.parser.add_argument("--is-training", action="store_true",
                            help="Whether to updates the running means and variances during the training.")
        self.parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                            help="Base learning rate for training with polynomial decay.")
        self.parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                            help="Base learning rate for discriminator.")
        self.parser.add_argument("--lambda-adv", type=float, default=LAMBDA_ADV,
                            help="lambda_adv for adversarial training.")
        self.parser.add_argument("--lambda-distill", type=float, default=LAMBDA_DISTILL,
                                 help="lambda_distill for knowledge distillation.")
        self.parser.add_argument("--momentum", type=float, default=MOMENTUM,
                            help="Whether to not restore last (FC) layers.")

        self.parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                            help="Number of training steps.")
        self.parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                            help="Number of training steps for early stopping.")
        self.parser.add_argument("--power", type=float, default=POWER,
                            help="Decay parameter to compute the learning rate.")
        self.parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                            help="Random seed to have reproducible results.")

        self.parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                            help="Save summaries and checkpoint every often.")
        self.parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                            help="Where to save snapshots of the model.")
        self.parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                            help="Regularisation parameter for L2-loss.")
        self.parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
        self.parser.add_argument("--tensorboard", help="choose whether to use tensorboard.", default=True)
        self.parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                            help="Path to the directory of log.")

        self.parser.add_argument("--set", type=str, default=SET,
                            help="choose adaptation set.")
        self.parser.add_argument("--gan", type=str, default=GAN,
                            help="choose the GAN objective.")

        self.parser.add_argument("--from-scratch", type=bool, default=FROM_SCRATCH)
        self.parser.add_argument("--source-only", action='store_true', default=SOURCE_ONLY)

        self.parser.add_argument("--dir-name", type=str, default=DIR_NAME)

        self.parser.add_argument("--pre-trained", type=str, default=PRE_TRAINED)

        self.parser.add_argument("--target", type=str, default=TARGET)

        self.parser.add_argument("--case", type=int, default=2)



