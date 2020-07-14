import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import imageio
import cv2

class SYNTHIADataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255, set='train'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set
        # for split in ["train", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, self.set, name)
            label_file = osp.join(self.root, "GT/LABELS/Stereo_Left/Omni_F/%s" % (name))
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:, :, 0]
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        if self.set == 'train':
            label = cv2.resize(label, self.crop_size, interpolation=cv2.INTER_NEAREST)

        label[label == 0] = 256
        label[label == 13] = 256
        label[label == 14] = 256
        label[label == 15] = 13
        label -= 1

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image / 255.0
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name


if __name__ == '__main__':
    dst = SYNTHIADataSet("/work/SYNTHIA-SEQS-04-SPRING", "./synthia_seqs_04_spring_list/train.txt", crop_size=(1024, 512),
                    scale=False, mirror=False, mean=np.array((0, 0, 0), dtype=np.float32))
    trainloader = data.DataLoader(dst, batch_size=1)
    all_imgs = 0.0
    for i, data in enumerate(trainloader):
        imgs, labels, shape, name = data
        all_imgs += imgs
        print(i, len(trainloader))
    img_mean = all_imgs.squeeze().view(3, -1).mean(dim=1) / len(trainloader)
    print(img_mean)
