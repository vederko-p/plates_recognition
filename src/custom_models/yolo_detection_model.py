
import torch
import pandas as pd


class DetectionModel:
    def __init__(self, weights_path: str):
        self.detector = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=weights_path)
        self.detector.eval()

    def __call__(self, image) -> pd.DataFrame:
        """Returns xmin, ymin, xmax, ymax, confidence"""
        target_cols = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence']
        detectoins = self.detector(image)
        results = detectoins.pandas().xyxy[0][target_cols]
        return results
