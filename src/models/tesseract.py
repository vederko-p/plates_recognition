import numpy as np
import cv2
import pytesseract


class TesseractOCR:

    @staticmethod
    def __build_tesseract_options(psm: int) -> str:
        alphanumeric = "АВЕКМНОРСТУХRUS0123456789"
        options = f"-c tessedit_char_whitelist={alphanumeric}"
        options += f" --psm {psm}"
        return options

    @staticmethod
    def __resize(img: np.ndarray,
                 width: int = None,
                 height: int = None,
                 inter: int = cv2.INTER_AREA) -> np.ndarray:
        dim = None
        (h, w) = img.shape[:2]

        if width is None and height is None:
            return img

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        resized = cv2.resize(img, dim, interpolation=inter)

        return resized

    @staticmethod
    def __preprocess(img: np.ndarray) -> np.ndarray:
        img = TesseractOCR.__resize(img, width=600)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def __init__(self, psm: int = 7):
        self.options = self.__build_tesseract_options(psm)

    def __call__(self, img: np.ndarray) -> str:
        img = self.__preprocess(img)
        plate_number = pytesseract.image_to_string(img, config=self.options, lang="rus")
        return plate_number
