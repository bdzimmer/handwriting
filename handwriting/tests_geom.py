# -*- coding: utf-8 -*-
"""
Unit tests for line utility functions.
"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import os
import pickle
import unittest

import cv2
import numpy as np

from handwriting import geom

VISUALIZE = True

RANDOM_SCALE = 10.0


class TestsGeom(unittest.TestCase):

    """Test geometry utility functions."""

    def test_point_line_distance(self):
        """tests for point line distance"""
        self.assertAlmostEqual(
            geom.point_to_line(0, 0, -1, -1, 1, 1),
            0.0)

        self.assertGreater(
            geom.point_to_line(0, 0, 0, -1, 1, 0),
            0.0)


    def test_compare_two_lines(self):
        """test function that compares two lines"""
        im, lines = _load_data()
        line1 = lines[5]
        line2 = _add_line_delta(
            line1, tuple(np.array(np.random.rand(4) * RANDOM_SCALE, dtype=np.int)))

        # distance from line to itself is zero
        # (this is basically an integration test for line_segment_within_image)
        score, distances, _ = geom.line_distance(line1, line1, im.shape)
        self.assertAlmostEqual(score, 0.0)

        score, distances, line_points = geom.line_distance(line1, line2, im.shape)

        if VISUALIZE:
            print("score:", score)
            print("distances:", distances)
            print("line_points:", line_points)
            _draw_line_on_image(im, line1, (0, 255, 0), 4, True)
            _draw_line_on_image(im, line2, (255, 0, 0), 4, True)
            for point in line_points:
                cv2.circle(im, (int(point[0]), int(point[1])), 12, (0, 0, 255), -1)
            _show(im)

        self.assertGreater(score, 0.0)


    def test_compare_line_lists(self):
        """test functions that compare two lists of lines"""

        im, lines = _load_data()
        lines = lines[0:5]
        lines_perturbed = [_add_line_delta(x, tuple(np.array(np.random.rand(4) * RANDOM_SCALE, dtype=np.int)))
                           for x in lines]

        score, scores, closest_idxs = geom.line_list_distance(lines, lines_perturbed, im.shape)

        if VISUALIZE:
            print("score:", score)
            print("scores:", scores)
            print("closest_idxs:", closest_idxs)
            for line in lines:
                _draw_line_on_image(im, line, (0, 255, 0), 4, True)
            for line in lines_perturbed:
                _draw_line_on_image(im, line, (255, 0, 0), 4, True)
            _show(im)

        self.assertGreater(score, 0.0)


def _load_data():
    """load an image and its associated lines file"""

    input_filename = "data/20170929_1.png"
    lines_filename = input_filename + ".lines.pkl"

    im = cv2.imread(input_filename)
    # im2 = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

    if os.path.exists(lines_filename):
        with open(lines_filename, "rb") as lines_file:
            preds = pickle.load(lines_file)
            lines = preds[0].result
    else:
        lines = []

    return im, lines


def _draw_line_on_image(im, line, color, width, draw_points):
    """draw a line on an image"""
    lf = geom.build_line_func(*line)
    cv2.line(im, lf(-10), lf(10), color, width)
    if draw_points:
        cv2.circle(im, (int(line[0]), int(line[1])), width * 3, color, -1)
        cv2.circle(im, (int(line[2]), int(line[3])), width * 3, color, -1)


def _show(im):
    """display an image"""
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", im)
    cv2.resizeWindow("image", int(im.shape[1] / 3.0), int(im.shape[0] / 3.0))
    cv2.waitKey(-1)


def _add_line_delta(line, delta):
    return (line[0] + delta[0], line[1] + delta[1], line[2] + delta[2], line[3] + delta[3])
