"""

Prepare datasets.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import gc
import random

import attr
import numpy as np

from handwriting import ml, util

VERBOSE = True


@attr.s
class PrepConfig:
    idxs = attr.ib(default=[])
    do_subsample = attr.ib(default=False)
    subsample_size = attr.ib(default=0)
    do_prep_balance = attr.ib(default=False)
    do_balance = attr.ib(default=False)
    balance_size = attr.ib(default=0)
    do_augment = attr.ib(default=False)
    augment_size = attr.ib(default=0)


def prepare(
        data_input,
        labels_input,
        do_subsample,
        subsample_size,
        do_prep_balance,
        do_balance,
        balance_size,
        do_augment,
        augment_size,
        augment_func,
        seed=0):

    """prepare a dataset from a raw list of data and labels
    by subsampling, balancing, and augmenting"""

    # WARNING! this function is not pure!
    # - destroys its inputs to save memory
    # - resets RNG seeds before each operation

    data = list(data_input)
    labels = list(labels_input)
    data_input.clear()
    labels_input.clear()

    if VERBOSE:
        print("\tunprepared data size:", util.mbs(data), "MiB")
        label_counts = ml.label_counts(labels)
        print("\tunprepared data group sizes:", label_counts[0])

    if do_subsample:
        if VERBOSE:
            print("\tsubsampling with size", subsample_size)
        np.random.seed(seed)
        random.seed(seed)
        data, labels = ml.subsample(data, labels, subsample_size)
        gc.collect()

    if do_prep_balance:
        if VERBOSE:
            print("\tpreparing for balance")
        np.random.seed(seed)
        random.seed(seed)
        data, labels = ml.prepare_balance(data, labels, balance_size)
        gc.collect()

    if do_subsample or do_prep_balance:
        # turn everything that's left into fresh arrays rather than memory views
        print("\tconverting views to arrays to free old data")
        data = [np.copy(x) for x in data]
        gc.collect()

    if do_balance:
        if VERBOSE:
            print("\tbalancing with size", balance_size)
        np.random.seed(seed)
        random.seed(seed)
        data, labels = ml.balance(data, labels, balance_size, augment_func)
        gc.collect()

    if do_augment:
        if VERBOSE:
            print("\taugmenting with size", augment_size)
        np.random.seed(seed)
        random.seed(seed)
        data, labels = ml.augment(data, labels, augment_size, augment_func)

    if VERBOSE:
        print("\tdone")

    return data, labels


def filter_labels(data, labels, keep_labels):
    """filter a dataset by label"""

    grouped = dict(ml.group_by_label(data, labels))

    grouped_filtered = {x: y for x, y in grouped.items() if x in keep_labels}
    return zip(*[
        (y, x[0]) for x in grouped_filtered.items() for y in x[1]])
