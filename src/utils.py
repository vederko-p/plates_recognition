
from cv2 import VideoCapture


def check_source(cap: VideoCapture):
    if not cap.isOpened():
        raise Exception('Error occurred  while opening video stream or file.')
