from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple

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


def text_detector(path) -> Tuple[list, list, np.ndarray]:
    image = cv2.imread(path)
    result = detector.ocr(image, rec=False, cls=True)
    res = None
    for idx in range(len(result)):
        res = result[idx]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sorted_coordinate = sorted(res, key=lambda coords: (coords[0][1], coords[0][0]))

    images = []
    point_list = []
    for coordinate in sorted_coordinate:
        points = np.array(coordinate, dtype=np.float32)
        crop = crop_image(image, points)
        crop = Image.fromarray(crop)
        images.append(crop)
        point_list.append(points)
    return images, point_list, image


def text_recognizer(images):
    s = recognizer.predict_batch(images)
    merged_s = ' '.join([w for w in s])
    del s
    return merged_s


def scanner(path):
    images, _, _ = text_detector(path)
    if len(images) > 0:
        s = text_recognizer(images)
        return s


def scanner_and_visualize(path):
    images, points, img = text_detector(path)
    sentences = []
    if len(images) > 0:
        sentences = recognizer.predict_batch(images)

    height = img.shape[0]
    width = img.shape[1]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(255)

    if len(sentences) > 0:
        for i, sentence in enumerate(sentences):
            rgb_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            bbox = np.array(points[i], dtype=np.int32)
            img = cv2.polylines(img, [bbox], isClosed=True, color=rgb_color, thickness=1)
            image = cv2.polylines(image, [bbox], isClosed=True, color=rgb_color, thickness=1)

            bbox_width = bbox[2][0] - bbox[0][0]
            bbox_height = bbox[2][1] - bbox[0][1]

            image = Image.fromarray(image)
            draw_text = ImageDraw.Draw(image)
            font_size = 24
            font = ImageFont.truetype('SVN-Arial Regular.ttf', size=font_size)
            text_width, text_height = draw_text.textsize(sentence, font=font)

            while text_width > bbox_width or text_height > bbox_height:
                font_size -= 1
                font = ImageFont.truetype('SVN-Arial Regular.ttf', size=font_size)
                text_width, text_height = draw_text.textsize(sentence, font=font)

            text_x = (bbox[0][0] + bbox[2][0] - text_width) // 2
            text_y = (bbox[0][1] + bbox[2][1] - text_height) // 2
            draw_text.text((text_x, text_y), sentence, fill='black', font=font)
            image = np.array(image)

        combine_image = np.hstack((img, image))
    else:
        combine_image = np.hstack((img, image))

    cv2.imshow('Image', combine_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
