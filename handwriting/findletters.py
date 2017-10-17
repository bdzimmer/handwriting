"""

Find positions of letters in images of words or a single word.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import numpy as np

from handwriting import util


def find_thresh_peaks(word_image):

    """find letter positions using thresholds and peaks in a KDE"""

    peak_sigma = 2 # 2
    mean_divisor = 1.0 # 1.0
    peak_percentile = 0 # 0

    # data = np.min(np.array(line_image[32:, :, 0], dtype=np.float), axis=0)
    data = np.std(np.array(word_image[32:, :, 0], dtype=np.float), axis=0)
    data_range = np.arange(0, word_image.shape[1])
    li_thresh = np.mean(data) / mean_divisor

    idxs = np.where(data < li_thresh)[0]

    if len(idxs) == 0:
        return []

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
    gaps = [0] + peak_idxs_filtered + [word_image.shape[0]]

    # convert gap locations to list of pairs
    positions = [x for x in list(zip(gaps[:-1], gaps[1:]))
                 if x[1] - x[0] > 0]

    return positions
