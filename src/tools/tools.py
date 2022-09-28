import cv2
import matplotlib.pyplot as plt

from os import listdir
from pathlib import Path
from random import sample


def debug_ocr(model, data: str, n: int = 10):
    """
    Randomly chooses n images from given data path and applies given model
    """
    data = Path(data)
    files = listdir(data)
    for imgfile in sample(files, n):
        img = cv2.imread(str(data / imgfile))
        res = model(img)
        plt.axis("off")
        plt.title(res if res else "couldn't recognize any")
        plt.imshow(img)
        plt.show()
