
import cv2

from utils import check_source
from configs.config_test_video import test_video_config


def process_video(cap: cv2.VideoCapture, display: bool = False):
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if display:
                cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    pass


def main(config: dict):
    v_cap = cv2.VideoCapture(config['video_filepath'])
    check_source(v_cap)
    process_video(v_cap, config['display_video'])
    pass


if __name__ == '__main__':
    main(test_video_config)
