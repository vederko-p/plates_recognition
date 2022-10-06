import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


CHARS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "A", "B", "C", "E", "H", "K", "M", "O", "P", "T",
         "X", "Y"]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


class LPRDataset(Dataset):
    def __init__(self, img_dir, img_size, lpr_max_len, preproc_fun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = img_size
        self.lpr_max_len = lpr_max_len
        if preproc_fun is not None:
            self.preproc_fun = preproc_fun
        else:
            self.preproc_fun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        img = cv2.imread(filename)
        height, width, _ = img.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            img = cv2.resize(img, self.img_size)
        img = self.preproc_fun(img)

        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            label.append(CHARS_DICT[c])

        if len(label) != 9:
            raise ValueError("Error label!")

        return img, label, len(label)

    def transform(self, img):
        img = img.astype("float32")
        img = np.transpose(img, (2, 0, 1))

        return img
