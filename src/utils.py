
import pandas as pd
from cv2 import VideoCapture


def check_source(cap: VideoCapture):
    if not cap.isOpened():
        raise Exception('Error occurred  while opening video stream or file.')


def get_seconds(timestamp: str) -> float:
    vals = timestamp.split('-')
    secs_code = [3600, 60, 1]
    return sum([float(v)*sc for v, sc in zip(vals, secs_code)])


def read_as_df(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as csv_file:
        csv_liens = csv_file.readlines()
    data_list = list(map(lambda line: line[:-1].split(',') , csv_liens))
    df = pd.DataFrame(data_list[1:], columns=data_list[0])
    df['timestamp'] = df['timestamp'].apply(lambda t: get_seconds(t))
    return df
