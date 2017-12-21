# -*- coding: utf-8 -*-
"""

Image processing and feature extraction functions.

"""

import cv2
import numpy as np


def pad_image(im, width, height, border=255):
    """pad char image in a larger image"""

    xoff = abs(int((im.shape[1] - width) / 2))
    yoff = abs(int((im.shape[0] - height) / 2))

    if width >= im.shape[1]:
        x_min_old = 0
        x_max_old = im.shape[1]
        x_min_new = xoff
        x_max_new = im.shape[1] + xoff
    else:
        x_min_old = xoff
        x_max_old = width + xoff
        x_min_new = 0
        x_max_new = width

    if height >= im.shape[0]:
        y_min_old = 0
        y_max_old = im.shape[0]
        y_min_new = yoff
        y_max_new = im.shape[0] + yoff
    else:
        y_min_old = yoff
        y_max_old = height + yoff
        y_min_new = 0
        y_max_new = height

    image_subset = im[y_min_old:y_max_old, x_min_old:x_max_old]
    new_bmp = np.ones((height, width, 3), dtype=np.uint8) * border
    new_bmp[y_min_new:y_max_new, x_min_new:x_max_new] = image_subset

    return new_bmp


def transform_random(image, trans_size, rot_size, scale_size):
    """apply a small random transformation to an image"""

    # TODO: make ranges of random numbers input parameters
    trans = np.random.rand(2) * trans_size - 0.5 * trans_size
    rot = np.random.rand(4) * rot_size - 0.5 * rot_size
    scale = 1.0 + np.random.rand(1)[0] * scale_size - 0.5 * scale_size

    x_size = image.shape[1]
    y_size = image.shape[0]

    trans_to_center = np.float32(
        [[1, 0, -x_size / 2.0],
         [0, 1, -y_size / 2.0],
         [0, 0, 1]])
    trans_from_center = np.float32(
        [[1, 0, x_size / 2.0],
         [0, 1, y_size / 2.0],
         [0, 0, 1]])
    trans_random = np.float32(
        [[1 + rot[0], 0 + rot[1], trans[0]],
         [0 + rot[2], 1 + rot[3], trans[1]],
         [0, 0, 1]])
    trans_scale = np.identity(3, dtype=np.float32) * scale

    tmat = np.dot(trans_from_center, np.dot(trans_scale, np.dot(trans_random, trans_to_center)))[0:2, :]

    image_new = cv2.warpAffine(
        image, tmat,
        (image.shape[1], image.shape[0]),
        borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", image_new)
    # cv2.waitKey()

    return image_new


def filter_cc(image):
    """find connected components in a threshold image and white out
    everything except the second largest"""

    # TODO: better way to select relevant components
    comp_filt = np.copy(image)
    gray = 255 - np.array(np.sum(image, axis=2) / 3.0, dtype=np.uint8)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = comps[1]
    sizes = comps[2][:, cv2.CC_STAT_AREA]

    # get index of second-largest component
    if len(sizes) > 1:
        second_largest_idx = np.argsort(sizes)[-2]
    else:
        second_largest_idx = np.argsort(sizes)[-1]

    # eliminate everything else
    for label_idx in range(len(sizes)):
        if label_idx != second_largest_idx:
            comp_filt[labels == label_idx] = 255

    # cv2.imshow("image", image)
    # cv2.imshow("gray", gray)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("comp_filt", comp_filt)
    # cv2.waitKey()

    return comp_filt


def align(image, x_align=True, y_align=True):
    """shift an image so the center of mass of the pixels is centered"""

    # TODO: this should just operate on grayscale

    gray = 255 - np.array(np.sum(image, axis=2) / 3.0, dtype=np.uint8)

    if x_align:
        x_size = image.shape[1]
        x_mean = np.sum(np.sum(gray, axis=0) * np.arange(x_size)) / np.sum(gray)
        x_shift = x_size / 2.0 - x_mean
    else:
        x_shift = 0.0

    if y_align:
        y_size = image.shape[0]
        y_mean = np.sum(np.sum(gray, axis=1) * np.arange(y_size)) / np.sum(gray)
        y_shift = y_size / 2.0 - y_mean
    else:
        y_shift = 0.0

    tmat = np.float32(
        [[1, 0, x_shift],
         [0, 1, y_shift]])
    new_image = cv2.warpAffine(
        image, tmat, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", new_image)
    # cv2.waitKey()

    return new_image


def grayscale(image):
    """convert RGB ubyte image to grayscale"""
    return np.sum(image, axis=2) / 3.0


def downsample(image, scale_factor):
    """downsample an image and unravel to create a feature vector"""

    feats = cv2.resize(
        image,
        (int(image.shape[0] * scale_factor),
         int(image.shape[1] * scale_factor)))
    return feats


def downsample_4(image):
    """create a feature vector from four downsampling amounts"""

    return downsample_multi(image, [0.4, 0.2, 0.1, 0.05])


def downsample_multi(image, scales):
    """create a feature vector from arbitrary downsampling amounts"""

    return np.hstack([np.ravel(downsample(image, x)) for x in scales])




def max_pool(im):
    """perform 2x2 max pooling"""

    return np.max(
        np.stack(
            (im[0::2, 0::2],
             im[0::2, 1::2],
             im[1::2, 0::2],
             im[1::2, 1::2]),
            axis=-1),
        axis=-1)


def max_pool_multi(image, ns):
    """perform multiple levels of max pooling and unravel
    to create a feature vector"""

    # TODO: move this to a higher level
    # image_gray = _grayscale(image)

    if 1 in ns:
        res = [image]
    else:
        res = []
    for n in range(2, max(ns) + 1):
        image = max_pool(image)
        if n in ns:
            res.append(image)
    return np.hstack([np.ravel(y) for y in res])


def column_ex(gray):

    """experimental feature - something like the center of mass of
    overlapping columns of the image"""

    width = 2
    # mul_mat = np.arange(y_size)[:, np.newaxis]
    # for some reason, it works a lot better to not divide by the sum of the
    # whole window but only the first column.
    mul_mat = np.linspace(0, 1, gray.shape[0])[:, np.newaxis]

    y_agg = np.array([(np.sum(gray[:, idx + width] * mul_mat) /
                       np.sum(gray[:, idx]))
                      for idx in range(gray.shape[1] - width)])
    y_agg[~np.isfinite(y_agg)] = 0.0

    res = np.hstack((y_agg, np.diff(y_agg)))

    return res


def extract_pos(pos, im, border=255):
    """extract a position (tuple of start and end) from an image"""

    # this is intended to have the correct logic to always return an image
    # of the width of the position even if it is off the edge of the image

    target_width = pos[1] - pos[0]
    extract = im[:, np.maximum(pos[0], 0):pos[1]]
    # print(cpos, extract.shape, im.shape)
    if extract.shape[1] < target_width:
        res = np.ones((im.shape[0], target_width, 3), dtype=np.ubyte) * border
        if pos[0] < 0:
            pr = (-pos[0], -pos[0] + extract.shape[1])
        else:
            pr = (0, extract.shape[1])
        # print(pr, flush=True)
        res[:, pr[0]:pr[1]] = extract
        return res
    else:
        res = extract
    return res
