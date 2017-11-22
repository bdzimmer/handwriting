"""

Find positions of characters in images of words or a single word.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

# TODO: rename to "findcharacters.py"


import cv2

import numpy as np

from handwriting import util


def find_thresh_peaks(word_image, peak_sigma=2.0, mean_divisor=1.0, peak_percentile=0):

    """find letter positions using thresholds and peaks in a KDE"""

    data = np.std(np.array(word_image[32:, :, 0], dtype=np.float), axis=0)

    data_range = np.arange(0, word_image.shape[1])
    li_thresh = np.mean(data) / mean_divisor

    idxs = np.where(data < li_thresh)[0]

    if len(idxs) == 0:
        return [(0, word_image.shape[1] - 1)]

    # smaller bandwidths get letters or pairs of letters!
    peak_idxs, peak_values = util.find_peak_idxs(
        idxs,
        data_range,
        peak_sigma)

    # print("peaks length:", len(peak_idxs))
    peak_thresh = np.percentile(peak_values, peak_percentile)

    peak_idxs_filtered = [x for x, y in zip(peak_idxs, peak_values)
                          if y >= peak_thresh]

    # print("filtered peaks length:", len(peak_idxs_filtered))
    gaps = [0] + peak_idxs_filtered + [word_image.shape[1] - 1]

    # convert gap locations to list of pairs
    return gaps_to_positions(gaps)


def find_classify(word_im, half_width, extract_char, classify_char_pos):
    """find character positions using a classifier"""

    char_poss = []
    run = []
    for x in range(2, word_im.shape[1] - 2, 1): # step 2
        test_range = (x - half_width, x + half_width)
        test_im = extract_char(test_range, word_im)
        # print("testing", test_range, test_im.shape)
        if classify_char_pos([test_im])[0]:
            run.append(x)
        else:
            if len(run) > 1:
                char_poss.append(int(np.mean(run)))
                run = []
    if len(run) > 1:
        char_poss.append(int(np.mean(run)))
        run = []

    return gaps_to_positions([0] + char_poss + [word_im.shape[1] - 1])


def find_classify_prob(
        word_im, half_width, extract_char, classify_char_pos, thresh):
    """find character positions using a classifier that returns a probability"""

    char_poss = []
    run = []
    for x in range(2, word_im.shape[1] - 2, 1): # step 2
        test_range = (x - half_width, x + half_width)
        test_im = extract_char(test_range, word_im)
        # disp_im = np.copy(word_im)
        # for idx in run:
        #     disp_im[:, idx[0], 0] = 255
        # disp_im[:, x] = (0, 0, 255)
        # cv2.imshow("test", disp_im)
        # cv2.waitKey()
        # print("testing", test_range, test_im.shape)
        prob = classify_char_pos([test_im])[0]
        if prob > thresh:
            run.append((x, prob))
        else:
            if len(run) > 1:
                char_poss.append(run[np.argmax([y[1] for y in run])][0])
                run = []
    if len(run) > 1:
        char_poss.append(run[np.argmax([y[1] for y in run])][0])
        run = []

    return gaps_to_positions([0] + char_poss + [word_im.shape[1] - 1])


def find_combine(word_im, extract_char, func1, func2):
    """find positions using one function, then find more positions inside
    those with another function."""

    def add(x, y):
        """helper"""
        return x[0] + y[0], x[0] + y[1]

    # init_poss =  findwords.find_conc_comp(word_im, merge=False)
    #  findletters.find_thresh_peaks(
    init_poss = func1(word_im)
    res = [add(x, y) for x in init_poss
            for y in func2(extract_char(x, word_im))]
    res_pos = positions_to_gaps(res) if len(res) > 0 else []
    return gaps_to_positions(
        [0] + res_pos + [word_im.shape[1] - 1])


def gaps_to_positions(gaps):
    """helper"""
    positions = [(x, y) for x, y in list(zip(gaps[:-1], gaps[1:]))
                 if y - x > 0]
    return positions


def positions_to_gaps(positions):
    """helper"""
    if len(positions) == 1:
        return [positions[0][0], positions[0][1]]
    else:
        return [positions[0][0]] + [int(0.5 * (x[1] + y[0]))
                                    for x, y in list(zip(positions[:-1], positions[1:]))] + [positions[-1][1]]


def position_distance(pos1, pos2):
    """calculate a distance score between two positions"""
    return np.square(pos2[0] - pos1[0]) + np.square(pos2[1] - pos1[1])


def jaccard_index(pos1, pos2):
    """calculate intersection over union for two positions"""
    # assumes that within each tuple, the values are in increasing order
    intersection = min(
        max(pos1[1] - pos2[0], 0),
        max(pos2[1] - pos1[0], 0))
    min_point = min(pos1[0], pos2[0])
    max_point = max(pos1[1], pos2[1])
    union = min(max_point - min_point, (pos2[1] - pos2[0]) + (pos1[1] - pos1[0]))
    res = intersection / union
    return res


def position_list_distance(
        positions_true, positions_test, dist_func):
    """calculate a distance score between two sets of character positions"""

    # for each true position, find the closest position in the test
    # positions.

    distances = np.zeros(len(positions_true))
    closest_idxs = np.zeros(len(positions_true), dtype=np.int)
    for idx, pos_true in enumerate(positions_true):
        # find the closest distance in the other set
        pos_true_to_positions_test = [dist_func(pos_true, x)
                                      for x in positions_test]
        closest_idx = np.argmin(pos_true_to_positions_test)
        closest_idxs[idx] = closest_idx
        distances[idx] = pos_true_to_positions_test[closest_idx]

    # return np.sqrt(np.mean(np.square(scores)))
    return distances, closest_idxs
