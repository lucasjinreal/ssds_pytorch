from __future__ import print_function

import sys
import os
import argparse
import numpy as np
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from lib.utils.config_parse import cfg_from_file
from lib.ssds_train import train_model


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
                        help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    train_model()


if __name__ == '__main__':
    train()
