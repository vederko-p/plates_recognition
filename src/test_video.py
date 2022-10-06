
import cv2

from csv_container import ContainerCSV
from our_utils import check_source
from our_utils import read_as_df
from configs.config_test_video import test_video_config
from metrics_video import tpr_fpr, rec_deviation

from custom_models.models_manager import ModelsManager
from custom_models.lazy_color_model import LazyColor
from custom_models.recognizer import PlateRecognizer
from custom_models.yolo_detection_model import DetectionModel


def process_video(
        cap: cv2.VideoCapture,
        model,
        bd,
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
            if result is not None:
                print(list(result))
                print('='*30)
            # bd.write_line(result)

            # quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


def test_model_outputs(gt_csv_filepath: str, model_csv_filepath: str) -> None:
    df_gt = read_as_df(gt_csv_filepath)
    df_st = read_as_df(model_csv_filepath)
    tpr, fpr = tpr_fpr(df_gt, df_st)
    rec_dev = rec_deviation(df_gt, df_st)
    metrics_csv = ContainerCSV(test_video_config['metrics_csv_directory_path'])
    metrics_csv.write_line({'tpr': tpr, 'fpr': fpr, 'rec_dev': rec_dev})
    metrics_csv.save()


def main(config: dict, model):
    # deal with model results (get from scratch or take already done):
    if config['st_filepath'] is None:
        csv_bd = ContainerCSV(test_video_config['output_csv_directory_path'])
        v_cap = cv2.VideoCapture(config['video_filepath'])
        check_source(v_cap)
        process_video(v_cap, model, csv_bd,
                      display=config['display_video'],
                      skip_frame_by_model=config['skip_frames_by_model'])
        model_results_filepath = csv_bd.save()
    else:
        model_results_filepath = config['st_filepath']

    # get metrics:
    test_model_outputs(config['gt_filepath'], model_results_filepath)


if __name__ == '__main__':
    cars_detector = DetectionModel('custom_models/cars_detection.pt')
    lp_detector = DetectionModel('custom_models/plates_detection.pt')
    ocr_model = PlateRecognizer(
        "../weights/Final_LPRNet_model.pth",
        "configs/train.yml")
    color_recognition = LazyColor(size=(10, 10))

    models_manager = ModelsManager(
        cars_detector,
        lp_detector,
        ocr_model,
        color_recognition)

    main(test_video_config, models_manager)
