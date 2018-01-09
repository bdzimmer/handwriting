# -*- coding: utf-8 -*-
"""

Manage training and test pages.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


SAMPLE_FILENAMES = (
    ["data/20170929_" + str(idx) + ".png.sample.pkl.1" # 0-4
     for idx in range(1, 6)] +
    ["data/20171120_" + str(idx) + ".png.sample.pkl.1" # 5-8
     for idx in range(1, 5)] +
    ["data/20171209_" + str(idx) + ".png.sample.pkl.1" # 9-12
     for idx in range(1, 5)])


def train_test_pages(test_idxs):

    """Get lists of training and test pages given idxs of test pages."""

    train_idxs = [
        idx for idx in range(len(SAMPLE_FILENAMES))
        if idx not in test_idxs]

    return pages(train_idxs, test_idxs)


def  pages(train_idxs, test_idxs):

    train_filenames = [
        SAMPLE_FILENAMES[idx]
        for idx in range(len(SAMPLE_FILENAMES))
        if idx in train_idxs]

    test_filenames = [
        SAMPLE_FILENAMES[idx]
        for idx in range(len(SAMPLE_FILENAMES))
        if idx in test_idxs]

    return train_filenames, test_filenames
