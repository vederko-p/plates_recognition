from yaml import load, Loader

import cv2
import torch
import numpy as np

from .lprnet import build_lprnet
from ..data import CHARS, CHARS_DICT


class PlateRecognizer:
    """
    Wrapper class for plates recognizer
    Used for inference
    """

    def __init__(self, weights_path: str, train_config: str):
        with open(train_config, "r") as f:
            self.train_config = load(f, Loader=Loader)

        self.lprnet = build_lprnet(lpr_max_len=self.train_config["lpr_max_len"],
                                   phase="eval",
                                   class_num=len(CHARS),
                                   dropout_rate=self.train_config["dropout_rate"])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lprnet.to(self.device)

        self.lprnet.load_state_dict(torch.load(weights_path))

    def __call__(self, img: np.ndarray):
        height, width, _ = img.shape
        img_size = self.train_config["img_size"]
        if height != img_size[1] or width != img_size[0]:
            img = cv2.resize(img, img_size)

        img = img.astype("float32")
        img = np.transpose(img, (2, 0, 1))
        img = np.array([img])
        img = torch.Tensor(img)

        img = img.to(self.device)

        prebs = self.lprnet(img)
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)

        to_chars = {v: k for k, v in CHARS_DICT.items()}

        pred = [to_chars[k] for k in preb_labels[0]]

        if pred[-3] == "0":
            pred = pred[:-3] + pred[-2:]

        return "".join(pred)
