import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
import argparse
from torchvision import models
from model.deeplab import Deeplab
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet
from ACE.model.model import Generator, VGG19features
# from model.model import Generator, VGG19features

MONITOR = True

BATCH_SIZE = 3
NUM_STEPS = 50000

SOURCE = 'GTA5'  # 'GTA5' or 'SYNTHIA'
if SOURCE == 'GTA5':
    DATA_DIRECTORY = '/work/GTA5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    NUM_CLASSES = 18
elif SOURCE == 'SYNTHIA':
    DATA_DIRECTORY = '/work/SYNTHIA'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    NUM_CLASSES = 13

NUM_TARGET = 1
if NUM_TARGET == 1:
    TARGET = 'CityScapes'
    DATA_DIRECTORY_TARGET = '/work/CityScapes'
    DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
    LR_GEN = 1e-3
    RESTORE_FROM_DEEPLAB = './pretrained_snapshot/GTA5_15000.pth'
    MEMORY = None
    GENERATOR_FILE = None
elif NUM_TARGET == 2:
    TARGET = 'IDD'
    DATA_DIRECTORY_TARGET = '/work/IDD_Segmentation'
    DATA_LIST_PATH_TARGET = './dataset/idd_list/train.txt'
    LR_GEN = 2e-4
    RESTORE_FROM_DEEPLAB = './snapshots/GTA5_CityScapes_50000.pth'
    MEMORY = 'CityScapes'
    GENERATOR_FILE = './snapshots/GTA5_CityScapes_50000_G.pth'
    MEMORY_FILE = './memory/' + MEMORY + '.pt'


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def lr_power(base_lr, iter, power, interval):
    return base_lr * pow(power, int(iter / interval))

def adjust_learning_rate_seg(optimizer, i_iter, lr, args):
    # lr = lr_poly(lr, i_iter, args.num_steps, 0.9)
    lr = lr_power(lr, i_iter, 0.9, 1000)
    optimizer.param_groups[0]['lr'] = lr

def adjust_learning_rate_gen(optimizer, i_iter, lr, args):
    # lr = lr_poly(lr, i_iter, args.num_steps, 0.9)
    lr = lr_power(lr, i_iter, 0.9, 1000)
    optimizer.param_groups[0]['lr'] = lr

def get_arguments():
    parser = argparse.ArgumentParser(description="ACE pre-training")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--target", type=str, default=TARGET)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--set", type=str, default='train')
    parser.add_argument("--save-pred-every", type=int, default=5000)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--learning-rate", type=float, default=LR_GEN)
    parser.add_argument("--num-target", type=int, default=NUM_TARGET)
    parser.add_argument("--random-seed", type=int, default=1338)

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

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    input_size = (512, 256)

    # cudnn.enabled = True

    # Create and load network
    # segmentation model
    model = Deeplab(args=args)
    saved_state_dict = torch.load(RESTORE_FROM_DEEPLAB)
    model.load_state_dict(saved_state_dict)
    model.train()
    model.to(device)

    # encoder (VGG19)
    original_encoder = models.vgg19(pretrained=True)
    pretrained_params = original_encoder.state_dict()
    encoder = VGG19features()
    new_params = encoder.state_dict().copy()
    for i in pretrained_params:
        # features.0.weight, classifier.0.weight
        # conv0.weight
        i_parts = i.split('.')
        if not i_parts[0] == 'classifier':
            new_params['conv' + '.'.join(i_parts[1:])] = pretrained_params[i]
    encoder.load_state_dict(new_params)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    encoder = encoder.to(device)

    # generator
    generator = Generator()
    if args.num_target > 1:
        generator.load_state_dict(torch.load(GENERATOR_FILE))
    generator.train()
    generator.to(device)

    optimizer_seg = optim.SGD(model.parameters(), lr=2.5e-4, momentum=0.9, weight_decay=5e-4)
    optimizer_gen = optim.SGD(generator.parameters(), lr=args.learning_rate, momentum=0.99, weight_decay=5e-5)
    cudnn.benchmark = False

    optimizer_seg.zero_grad()
    optimizer_gen.zero_grad()

    # dataloaders
    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                        crop_size=input_size, ignore_label=args.ignore_label,
                        set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    elif args.source == 'SYNTHIA':
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                           crop_size=input_size, ignore_label=args.ignore_label,
                           set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    else:
        raise NotImplementedError('Unavailable source domain')

    trainloader_iter = enumerate(trainloader)

    if args.target == 'CityScapes':
        targetloader = torch.utils.data.DataLoader(cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                                     max_iters=args.num_steps,
                                                                     crop_size=input_size,
                                                                     ignore_label=args.ignore_label,
                                                                     set=args.set),
                                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    elif args.target == 'IDD':
        targetloader = torch.utils.data.DataLoader(IDDDataSet(args.data_dir_target, args.data_list_target,
                                                              max_iters=args.num_steps,
                                                              crop_size=input_size,
                                                              ignore_label=args.ignore_label,
                                                              set=args.set),
                                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
    else:
        raise NotImplementedError("Unavailable target domain")
    targetloader_iter = enumerate(targetloader)

    if args.num_target > 1:
        statistic_memory_before = torch.load(MEMORY_FILE)
        statistic_memory_before = torch.unique(statistic_memory_before, dim=0).cpu().numpy()
        random_indices = np.random.choice(statistic_memory_before.shape[0], size=100, replace=False)
        statistic_memory_before = torch.FloatTensor(statistic_memory_before[random_indices, :]).to(device)

    # Define loss functions
    seg_loss = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # Training
    statistics_memory = None
    for i_iter in range(args.num_steps):

        loss_gen_value = 0
        loss_seg_value = 0

        optimizer_seg.zero_grad()
        optimizer_gen.zero_grad()

        adjust_learning_rate_seg(optimizer_seg, i_iter, 2.5e-4, args)
        adjust_learning_rate_gen(optimizer_gen, i_iter, args.learning_rate, args)

        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        _, batch = targetloader_iter.__next__()

        images_target, _, _ = batch
        images_target = images_target.to(device)

        # extract features
        z_source = encoder(images)
        z_target = encoder(images_target)

        # calculate feature statistics
        mean_z_s = z_source.view(z_source.shape[0], z_source.shape[1], -1).mean(dim=2, keepdim=True).unsqueeze(dim=3)
        std_z_s = z_source.view(z_source.shape[0], z_source.shape[1], -1).std(dim=2, keepdim=True).unsqueeze(dim=3)
        mean_z_t = z_target.view(z_target.shape[0], z_target.shape[1], -1).mean(dim=2, keepdim=True).unsqueeze(dim=3)
        std_z_t = z_target.view(z_target.shape[0], z_target.shape[1], -1).std(dim=2, keepdim=True).unsqueeze(dim=3)
        if args.num_target > 1:
            mean_z_before = statistic_memory_before[random.randint(0, 99)][:512].unsqueeze(1).unsqueeze(2).unsqueeze(0)
            std_z_before = statistic_memory_before[random.randint(0, 99)][512:].unsqueeze(1).unsqueeze(2).unsqueeze(0)

        # store to memory
        statistics_t = torch.cat([mean_z_t.squeeze().unsqueeze(0), std_z_t.squeeze().unsqueeze(0)], dim=1)
        if statistics_memory == None:
            statistics_memory = statistics_t
        else:
            if len(statistics_memory) < 1000:
                statistics_memory = torch.cat([statistics_memory, statistics_t], dim=0)

        # AdaIN
        z_hat = std_z_t * ((z_source - mean_z_s) / (std_z_s + 1e-6)) + mean_z_t
        if args.num_target > 1:
            z_hat_before = std_z_before * ((z_source - mean_z_s) / (std_z_s + 1e-6)) + mean_z_before

        # calculate loss_gen
        z_tilda = encoder(generator(z_hat))
        mean_z_tilda = z_tilda.view(z_tilda.shape[0], z_tilda.shape[1], -1).mean(dim=2, keepdim=True).unsqueeze(dim=3)
        std_z_tilda = z_tilda.view(z_tilda.shape[0], z_tilda.shape[1], -1).std(dim=2, keepdim=True).unsqueeze(dim=3)
        if args.num_target > 1:
            z_tilda_before = encoder(generator(z_hat_before))
            mean_z_tilda_before = z_tilda_before.view(z_tilda_before.shape[0], z_tilda_before.shape[1], -1)\
                .mean(dim=2, keepdim=True).unsqueeze(dim=3)
            std_z_tilda_before = z_tilda_before.view(z_tilda_before.shape[0], z_tilda_before.shape[1], -1)\
                .std(dim=2, keepdim=True).unsqueeze(dim=3)

        loss_gen = torch.norm(z_tilda - z_hat) / (32 * 64) \
                   + 0.01 * torch.norm(mean_z_tilda - mean_z_t) \
                   + 0.01 * torch.norm(std_z_tilda - std_z_t)
        if args.num_target > 1:
            loss_gen += torch.norm(z_tilda_before - z_hat_before) / (32 * 64) \
                        + 0.01 * torch.norm(mean_z_tilda_before - mean_z_before) \
                        + 0.01 * torch.norm(std_z_tilda_before - std_z_before)
        if MONITOR:
            print(torch.norm(z_tilda - z_hat) / (32 * 64))
            print(0.01 * torch.norm(mean_z_tilda - mean_z_t))
            print(0.01 * torch.norm(std_z_tilda - std_z_t))
        loss_gen /= args.batch_size
        loss_gen_value += loss_gen.item()

        loss_gen.backward()
        optimizer_gen.step()

        for param in generator.parameters():
            param.requires_grad = False

        # calculate loss_seg
        x_hat = generator(z_hat)
        pred_hat = model(x_hat, input_size)
        if args.num_target > 1:
            x_hat_before = generator(z_hat_before)
            pred_hat_before = model(x_hat_before, input_size)
        pred = model(images, input_size)

        loss_kl = (((F.softmax(pred_hat, dim=1) * F.log_softmax(pred_hat, dim=1))
                   - (F.softmax(pred_hat, dim=1) * F.log_softmax(pred, dim=1))) / (args.batch_size * 512 * 256)).sum()

        loss_seg = seg_loss(pred, labels) + seg_loss(pred_hat, labels) + loss_kl
        if args.num_target > 1:
            loss_kl_before = (((F.softmax(pred_hat_before, dim=1) * F.log_softmax(pred_hat_before, dim=1))
                   - (F.softmax(pred_hat_before, dim=1) * F.log_softmax(pred, dim=1))) / (args.batch_size * 512 * 256)).sum()
            loss_seg += loss_kl_before + seg_loss(pred_hat_before, labels)
        if MONITOR:
            print(seg_loss(pred, labels))
            print(seg_loss(pred_hat, labels))
            print(loss_kl)

        loss_seg_value += loss_seg.item()

        loss_seg.backward()
        optimizer_seg.step()

        for param in generator.parameters():
            param.requires_grad = True

        print('iter = {0:8d}/{1:8d}, loss_seg = {2:.3f} loss_gen = {3:.3f}'
              .format(i_iter, args.num_steps, loss_seg, loss_gen))

        if not os.path.exists('./snapshots'):
            os.makedirs('./snapshots')
        if not os.path.exists('./memory'):
            os.makedirs('./memory')

        if i_iter >= args.num_steps - 1:
            if MEMORY == None:
                torch.save(model.state_dict(), osp.join('./snapshots', args.source + '_' + args.target + '_'
                                                        + str(args.num_steps) + '.pth'))
                torch.save(generator.state_dict(), osp.join('./snapshots', args.source + '_' + args.target + '_'
                                                        + str(args.num_steps) + '_G.pth'))
            else:
                torch.save(model.state_dict(), osp.join('./snapshots', args.source + '_' + MEMORY +
                                                        '_' + args.target + '_'
                                                        + str(args.num_steps) + '.pth'))
                torch.save(generator.state_dict(), osp.join('./snapshots', args.source + '_' + MEMORY +
                                                        '_' + args.target + '_'
                                                        + str(args.num_steps) + '_G.pth'))
            torch.save(statistics_memory, osp.join('./memory', args.target + '.pt'))
            break
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            if MEMORY == None:
                torch.save(model.state_dict(), osp.join('./snapshots', args.source + '_' + args.target + '_'
                                                        + str(i_iter) + '.pth'))
                torch.save(generator.state_dict(), osp.join('./snapshots', args.source + '_' + args.target + '_'
                                                        + str(i_iter) + '_G.pth'))
            else:
                torch.save(model.state_dict(), osp.join('./snapshots', args.source + '_' + MEMORY +
                                                        '_' + args.target + '_'
                                                        + str(i_iter) + '.pth'))
                torch.save(generator.state_dict(), osp.join('./snapshots', args.source + '_' + MEMORY +
                                                        '_' + args.target + '_'
                                                        + str(i_iter) + '_G.pth'))

if __name__ == "__main__":
    main()