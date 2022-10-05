
import utils


class ModelsManager:
    def __init__(self, car_detection_model, lp_detector_model,
                 ocr_model, color_model):
        self.car_detector = car_detection_model
        self.lp_detector = lp_detector_model
        self.ocr_model = ocr_model
        self.color_model = color_model

    def __call__(self, image):
        cars_detections = self.car_detector(image)
        cars_crops = self.get_crops_from_pd(image, cars_detections)
        plates_recognized = self.handle_plates(cars_crops)
        color_predicts = self.handle_colors(cars_crops)
        return zip(plates_recognized, color_predicts)

    def handle_plates(self, cars_crops: list) -> list:
        plates_crops = []
        for car in cars_crops:
            plate_detection = self.lp_detector(car)
            if plate_detection.shape[0] < 1:
                plates_crops.append(None)
            else:
                plate_detect_coords = plate_detection.iloc[0].tolist()[:-1]
                plate_crop = utils.get_crop_from_np(car, *plate_detect_coords)
                plates_crops.append(plate_crop)
        plates_recognized = []
        for plate in plates_crops:
            plates_recognized.append(
                None if plate is None else self.ocr_model(plate))
        return plates_recognized

    def handle_colors(self, cars_crops: list) -> list:
        return [self.color_model(utils.np2tensor(car)) for car in cars_crops]

    @staticmethod
    def get_crops_from_pd(image, pd_detections) -> list:
        res = []
        for _, row in pd_detections.iterrows():
            crop = utils.get_crop_from_np(image, *row.tolist()[:-1])
            res.append(crop)
        return res
