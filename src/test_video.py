
import cv2

from models.fake_model import FakeModel

from utils import check_source
from configs.config_test_video import test_video_config


def process_video(
        cap: cv2.VideoCapture,
        model,
        display: bool = False,
        skip_frame_by_model: int = 0
) -> None:
    skip_frame_by_model += 1
    frames_counter = 0
    while cap.isOpened():
        # get frame
        ret, frame = cap.read()
        frames_counter += 1
        result = None

        if ret:
            # display
            if display:
                cv2.imshow('Frame', frame)

            # model
            if not frames_counter % skip_frame_by_model:
                result = model(frame)

            # process result
            print(result)

            # quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


def main(config: dict, model):
    v_cap = cv2.VideoCapture(config['video_filepath'])
    check_source(v_cap)
    process_video(v_cap, model,
                  display=config['display_video'],
                  skip_frame_by_model=config['skip_frames_by_model'])


if __name__ == '__main__':
    my_model = FakeModel()
    main(test_video_config, my_model)
