# -*- coding: utf-8 -*-
'''
YiTian - Fingering Corrector Module

Copyright (c) 2025 Zhang Zhewei (Liyue-Wei)
Licensed under the GNU GPL v3.0 License. 
'''

from multiprocessing import shared_memory
from extmodules import shm_cfg
import numpy as np
import cv2

class FingeringCorrector:
    def __init__(self):
        self.shm_frame = None
        self.shm_result = None
        self.WIDTH = shm_cfg.WIDTH
        self.HEIGHT = shm_cfg.HEIGHT

    def key_map_calibration(self):
        anchor = ['q', 'p', 'z', 'm']