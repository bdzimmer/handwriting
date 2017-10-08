"""

Find words in an image of a single line of text.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import numpy as np


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
