from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
from PIL import Image


def get_cropped_size(src_points):
    x1, y1 = src_points[0]
    x2, y2 = src_points[1]
    x3, y3 = src_points[2]
    x4, y4 = src_points[3]

    width = int(max(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)))
    height = int(max(np.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2), np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)))

    return width, height


def crop_image(img, src_points):
    width, height = get_cropped_size(src_points)

    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    warped_img = cv2.warpPerspective(img, matrix, (width, height))

    return warped_img


config = Cfg.load_config_from_name("vgg_seq2seq")
config['device'] = "cpu"

detector = PaddleOCR(use_angle_cls=True)
recognizer = Predictor(config)


def text_detector(path) -> list:
    image = cv2.imread(path)
    result = detector.ocr(image, rec=False, cls=True)
    res = None
    for idx in range(len(result)):
        res = result[idx]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sorted_coordinate = sorted(res, key=lambda coords: (coords[0][1], coords[0][0]))

    images = []
    for coordinate in sorted_coordinate:
        points = np.array(coordinate, dtype=np.float32)
        crop = crop_image(image, points)
        crop = Image.fromarray(crop)
        images.append(crop)

    return images


def text_recognizer(images):
    s = recognizer.predict_batch(images)
    merged_s = ' '.join([w for w in s])
    del s
    return merged_s


def scanner(path):
    images = text_detector(path)
    if len(images) > 0:
        s = text_recognizer(images)
        return s




