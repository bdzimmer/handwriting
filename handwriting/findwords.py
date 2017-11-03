"""

Find words in an image of a single line of text.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import cv2
import numpy as np


VISUALIZE = False


def find_thresh(line_image):

    """find words using a simple threshold and sliding window."""

    # subset to section of line that contains words

    # find indicators of being in a word
    data = np.min(np.array(line_image[32:, :, 0], dtype=np.float), axis=0)
    li_thresh = np.mean(data) / 2.0
    in_word_bool = data < li_thresh

    # eliminate stray indicators
    half_window = 16
    window_thresh = 8
    in_word_bool_filt = np.zeros(in_word_bool.shape, dtype=np.bool)
    for idx in range(line_image.shape[1]):
        in_word_bool_filt[idx] = np.sum(in_word_bool[(idx - half_window):(idx + half_window)]) > window_thresh

    # in_word = np.where(in_word_bool_filt)[0]

    # find start and end of true sections of in_word
    positions = []
    pair = [0, 0]
    p_in_word = False
    for idx, value in enumerate(in_word_bool_filt):
        if not p_in_word and value:
            pair[0] = idx
            p_in_word = True
        elif p_in_word and not value:
            pair[1] = idx - 1
            p_in_word = False
            positions.append((pair[0], pair[1]))

    return positions


def find_conc_comp(line_image, merge=True, merge_tol=8):

    """find words using connected components."""

    gray = 255 - np.array(np.sum(line_image, axis=2) / 3.0, dtype=np.uint8)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # labels = comps[1]
    sizes = comps[2][:, cv2.CC_STAT_AREA]

    # find indices of all connected components except largest
    comp_idxs = np.argsort(sizes)[:-1]

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey()

    # convert to position tuples
    def stat_to_pos(idx):
        """convert component index to position tuple"""
        left = comps[2][idx, cv2.CC_STAT_LEFT]
        right = left + comps[2][idx, cv2.CC_STAT_WIDTH] - 1
        return (left, right)

    positions = [stat_to_pos(idx) for idx in comp_idxs]

    # filter out negative or zero width slides
    positions = [x for x in positions if x[1] - x[0] > 1 and x[0] >= 0 and x[1] >= 0]
    positions = sorted(positions, key=lambda x: x[0])

    final_positions = []

    if merge:
        for x in positions:
            merged = False
            for idx, y in enumerate(final_positions):
                if x[0] >= y[0] and x[0] <= y[1]:
                    if x[1] > y[1]:
                        final_positions[idx] = (y[0], x[1])
                    merged = True
                elif x[1] <= y[1] and x[1] >= y[0]:
                    if x[0] < y[0]:
                        final_positions[idx] = (x[0], y[1])
                    merged = True
                elif x[0] <= y[0] and x[1] >= y[1]:
                    final_positions[idx] = x
                    merged = True
                elif abs(x[1] - y[0]) < merge_tol:
                    final_positions[idx] = (x[0], y[1])
                    merged = True
                elif abs(y[1] - x[0]) < merge_tol:
                    final_positions[idx] = (y[0], x[1])
                    merged = True
                if merged:
                    break
            if not merged:
                final_positions.append(x)
    else:
        final_positions = positions

    final_positions = [x for x in final_positions if x[1] - x[0] > 1 and x[0] >= 0 and x[1] >= 0]
    final_positions = sorted(final_positions, key=lambda x: x[0])

    if VISUALIZE:
        cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
        cv2.imshow("thresh", thresh)
        cv2.namedWindow("char", cv2.WINDOW_NORMAL)
        for x in final_positions:
            print(x, flush=True)
            cv2.imshow("char", thresh[:, x[0]:x[1]])
            cv2.waitKey()
        print(final_positions)

    return final_positions
