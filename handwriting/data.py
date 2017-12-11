# -*- coding: utf-8 -*-
"""

Manage training and test pages.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


def train_test_pages(test_idxs):

    """Get lists of training and test pages given idxs of test pages."""

    sample_filenames = (
        ["data/20170929_" + str(idx) + ".png.sample.pkl.1"
         for idx in range(1, 6)] +
        ["data/20171120_" + str(idx) + ".png.sample.pkl.1"
         for idx in range(1, 5)] +
        ["data/20171209_" + str(idx) + ".png.sample.pkl.1"
         for idx in range(1, 2)])

    train_filenames = [
        sample_filenames[idx]
        for idx in range(len(sample_filenames))
        if idx not in test_idxs]

    test_filenames = [
        sample_filenames[idx]
        for idx in range(len(sample_filenames))
        if idx in test_idxs]

    return train_filenames, test_filenames
