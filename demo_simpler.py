"""
inference on trained models

with only provide a simple config file

"""
import sys
import os
import numpy as np
import cv2

from lib.ssds import ObjectDetector
from lib.utils.config_parse import cfg_from_file
import argparse

img_f = 'experiments/person.jpg'


def parse_args():
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('--cfg', default='experiments/cfgs/fssd_lite_mobilenetv2_train_voc.yml',
                        help='the address of optional config file')
    args = parser.parse_args()
    return args


def predict():
    args = parse_args()

    cfg_from_file(args.cfg)

    detector = ObjectDetector()

    img = cv2.imread(img_f)

    _labels, _scores, _coords = detector.predict(img)
    print('labels: {}\nscores: {}\ncoords: {}'.format(_labels, _scores, _coords))


if __name__ == '__main__':
    predict()