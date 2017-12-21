# -*- coding: utf-8 -*-
"""

Unit tests for image processing and feature extraction functions.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import unittest

import cv2
import numpy as np

from handwriting import improc

VISUALIZE = False


class TestsImproc(unittest.TestCase):

    """Unit tests for image processing and feature extraction functions."""

    def test_pad_image(self):
        """test function to pad an image"""

        # image is smaller in both dimensions
        image = np.ones((8, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = improc.pad_image(image, 16, 16)
        self.assertEqual(image_padded.shape, (16, 16, 3))

        if VISUALIZE:
            image_both = np.zeros((16, 32, 3), dtype=np.uint8)
            image_both[0:image.shape[0], 0:image.shape[1], :] = image
            image_both[0:image_padded.shape[0], 16:(image_padded.shape[1] + 16)] = image_padded
            cv2.namedWindow("padding", cv2.WINDOW_NORMAL)
            cv2.imshow("padding", image_both)
            cv2.waitKey()

        # image is larger in both dimensions
        image = np.ones((16, 16, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = improc.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is larger in x dimension
        image = np.ones((8, 16, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = improc.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is larger in y dimension
        image = np.ones((16, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = improc.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is same in both dimensions
        image = np.ones((8, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = improc.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))
