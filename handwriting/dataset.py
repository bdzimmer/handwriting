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
    # - resets RNG seeds after each operation

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
        data, labels = ml.subsample(data, labels, subsample_size)
        gc.collect()
        np.random.seed(seed)
        random.seed(seed)

    if do_prep_balance:
        if VERBOSE:
            print("\tpreparing for balance")
        data, labels = ml.prepare_balance(data, labels, balance_size)
        gc.collect()
        np.random.seed(seed)
        random.seed(seed)

    if do_balance:
        if VERBOSE:
            print("\tbalancing with size", balance_size)
        data, labels = ml.balance(data, labels, balance_size, augment_func)
        np.random.seed(seed)
        random.seed(seed)

    if do_augment:
        if VERBOSE:
            print("\taugmenting with size", augment_size)
        data, labels = ml.augment(data, labels, augment_size, augment_func)
        np.random.seed(seed)
        random.seed(seed)

    if VERBOSE:
        print("\tdone")

    return data, labels
