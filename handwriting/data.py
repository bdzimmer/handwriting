# -*- coding: utf-8 -*-
"""

Manage training and test pages.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.


SAMPLE_FILENAMES = (
    ["data/20170929_" + str(idx) + ".png.sample.pkl.1"  # 0-4
     for idx in range(1, 6)] +
    ["data/20171120_" + str(idx) + ".png.sample.pkl.1"  # 5-8
     for idx in range(1, 5)] +
    ["data/20171209_" + str(idx) + ".png.sample.pkl.1"  # 9-12
     for idx in range(1, 5)] +
    ["data/20180109_" + str(idx) + ".png.sample.pkl.1"  # 13-15
     for idx in range(1, 4)] +
    ["data/20180224_1.png.sample.pkl.1"]  # 16
)


def pages(idxs):
    """get lists of filenames given idxs of pages"""
    return [SAMPLE_FILENAMES[idx] for idx in idxs]
    return filenames
