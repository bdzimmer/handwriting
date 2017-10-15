"""

Find lines of text in an image.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

from handwriting import geom, util

VISUALIZE = False


def find(image):

    """find lines of text in an image"""

    blur_size = 8 # 4

    imf = np.array(image, dtype=np.float)
    b = imf[:, :, 0]
    g = imf[:, :, 1]
    r = imf[:, :, 2]
    gray = r + g + b

    gray = cv2.normalize(gray, gray, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    gray = gray * gray
    gray = cv2.normalize(gray, gray, alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
    gray = np.uint8(gray)

    if VISUALIZE:
        print(np.min(gray), np.max(gray))
        _show(gray, "gray", 1)

    edges = 255 - gray
    edges = cv2.blur(edges, (blur_size, blur_size))

    if VISUALIZE:
        _show(edges, "edges", 1)

    ret, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if VISUALIZE:
        _show(thresh, "thresh", 1)

    comp_filt = np.copy(thresh)
    connectivity = 4
    comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = comps[1]
    sizes = comps[2][:, cv2.CC_STAT_AREA]
    for label_idx, size in enumerate(sizes):
        if size > 5000:
            comp_filt[labels == label_idx] = 0

    if VISUALIZE:
        _show(comp_filt, "comp_filt", 1)

    edge_lines, final_lines = _lines_in_edge_image(comp_filt)

    disp_im = np.copy(image)
    l_width = max(image.shape[0], image.shape[1])

    if True:
        for line in edge_lines:
            lf = geom.build_line_func(*[int(x) for x in line])

            cv2.line(disp_im, lf(-10), lf(10), (255, 0, 0, 80), 4)

    for line in final_lines:
        lf = geom.build_line_func(*[int(x) for x in line])
        cv2.line(disp_im, lf(0), lf(l_width), (0, 255, 0), 4)

    if VISUALIZE:
        _show(disp_im, "result", 1)

    return edge_lines, final_lines


def _show(im, title, wait_key, sd=1):
    """helper"""
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, im)
    cv2.resizeWindow(title, int(im.shape[1] / sd), int(im.shape[0] / sd))
    if wait_key >= 0:
        key = cv2.waitKey(wait_key)
        if key == 27:
            sys.exit()


def _lines_in_edge_image(edge_image):
    """find lines in edges"""

    r_sigma = 8.0
    theta_sigma = 0.02

    thresh = 40 # 40!
    min_line_length = 10 # 20
    max_line_gap = 2 # 25 1

    filter_lines = True

    if True:
        lines = cv2.HoughLinesP(
            edge_image, 1, np.pi / 180.0, thresh, min_line_length, max_line_gap)
        if lines is None:
            return [], []
        lines = np.array([x[0] for x in lines])
        lines_polar = np.array([geom.points_to_polar(*x) for x in lines])
    else:
        lines_polar = cv2.HoughLines(
            edge_image, 1, np.pi / 180.0, 100)
        print(lines_polar)
        if lines_polar is None:
            return [], []
        lines = np.array([geom.polar_to_points(*x) for x in lines_polar])

    print("line count:", len(lines_polar))

    if False:
        # plot histograms of r and theta values
        plt.figure(1)
        plt.hist(lines_polar[:, 0], 50, facecolor="blue")
        plt.xlabel("R")
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.figure(2)
        plt.hist(lines_polar[:, 1], 50, facecolor="blue")
        plt.xlabel("Theta")
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.figure(3)
        plt.scatter(lines_polar[:, 0], lines_polar[:, 1])

        plt.show(block=False)

    if filter_lines:
        # filter by theta within pi / 100 of mean theta
        # theta_mean = np.mean(lines_polar[:, 1])
        # print("initial theta mean:", theta_mean)

        if True:
            # assumes that the paper is near portrait orientation
            theta_mean = 0.0
        else:
            # doesn't make assumptions about paper orientation, but not
            # sure if this really works
            th_range = np.arange(
                np.min(lines_polar[:, 1]) - 0.04,
                np.max(lines_polar[:, 1]) + 0.04,
                theta_sigma)
            peak_idxs, peak_values = util.find_peak_idxs(lines_polar[:, 1], th_range, 0.1)
            theta_idx = peak_idxs[np.argmax(peak_values)]
            theta_mean = th_range[theta_idx]

        print("theta mean:", theta_mean * 180 / np.pi, "degrees")

        keep_idxs = np.abs(lines_polar[:, 1] - theta_mean) < np.pi / 100.0
        lines = lines[keep_idxs, :]
        lines_polar = lines_polar[keep_idxs, :]

        if False:
            # plot histograms of r and theta values
            plt.figure(1)
            plt.hist(lines_polar[:, 0], 50, facecolor="blue")
            plt.xlabel("R")
            plt.ylabel('Frequency')
            plt.grid(True)

            plt.figure(2)
            plt.hist(lines_polar[:, 1], 50, facecolor="blue")
            plt.xlabel("Theta")
            plt.ylabel('Frequency')
            plt.grid(True)

            plt.show(block=False)

    print("line count after filtering:", len(lines))

    if len(lines) == 0:
        return [], []

    theta_mean = np.mean(lines_polar[:, 1])

    print("theta mean before adjustment:", theta_mean)

    # TODO: not sure if this logic is exactly what I want
    if theta_mean > 0.5 * np.pi:
        theta_mean = theta_mean - np.pi
    if theta_mean < -0.75 * np.pi:
        theta_mean = theta_mean + np.pi

    print("theta mean:", theta_mean)

    r_range = np.arange(0, int(np.max(lines_polar[:, 0])))
    # util.VISUALIZE = True
    peak_idxs, _ = util.find_peak_idxs(lines_polar[:, 0], r_range, r_sigma)

    print("final line count:", len(peak_idxs))

    final_lines = []

    rs = [r_range[idx] for idx in peak_idxs]

    if True and len(rs) > 1:

        # repeatedly remove lines where the diff before
        # it is less than 0.75 of the mean diff
        # this doesn't work so great, since it can remove valid
        # lines if there is an extra line after an empty space

        rs_filt = [x for x in rs]
        while True:
            rs_diff = np.diff(rs_filt)
            rs_diff_mean = np.mean(rs_diff)
            # print(rs_diff)
            unchanged = True
            for idx, dval in enumerate(rs_diff):
                if dval < 0.75 * rs_diff_mean:
                    # print(idx + 1, dval, 0.75 * rs_diff_mean)
                    rs_filt.pop(idx + 1)
                    unchanged = False
                    break
            if unchanged:
                break
    else:
        rs_filt = rs

    for r in rs_filt:
        pts = geom.polar_to_points(r, theta_mean)
        final_line = geom.line_segment_within_image(
            pts, edge_image.shape)
        final_lines.append([int(x) for x in final_line])

    return lines, final_lines
