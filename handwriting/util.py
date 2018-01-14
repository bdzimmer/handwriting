"""

General utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import dill
import numpy as np
from sklearn.neighbors import KernelDensity

dill.settings["recurse"] = True


def load(input_filename):
    """unpickle an object from a file"""
    with open(input_filename, "rb") as input_file:
        res = pickle.load(input_file)
    return res


def save(obj, output_filename):
    """pickle an object to a file"""
    with open(output_filename, "wb") as output_file:
        pickle.dump(obj, output_file)


def load_dill(input_filename):
    """undill an object from a file"""
    with open(input_filename, "rb") as input_file:
        res = dill.load(input_file)
    return res


def save_dill(obj, output_filename):
    """dill an object to a file"""
    with open(output_filename, "wb") as output_file:
        dill.dump(obj, output_file)


def patch_image(bmps, width=16, height=16):

    """combine equally sized smaller images into a larger image"""

    if not bmps:
        return np.zeros((16, 16), dtype=np.uint8)

    # TODO: get rid of default values for width and height
    patch_height = bmps[0].shape[0]
    patch_width = bmps[0].shape[1]
    if len(bmps[0].shape) == 2:
        grayscale = True
    else:
        grayscale = False
    res = np.zeros(
        (height * patch_height, width * patch_width, 3),
        dtype=np.uint8)

    for idx in range(min(len(bmps), width * height)):
        col = (idx % width) * patch_width
        row = int(idx / width) * patch_height
        bmp = bmps[idx]
        if grayscale:
            bmp = np.expand_dims(bmp, 2).repeat(3, 2)
        res[row:(row + patch_height), col:(col + patch_width), :] = bmps[idx]

    return res


def find_peak_idxs(data, data_range, bandwidth, visualize=False):

    """find locations of peaks in a KDE"""

    # build 1D KDE of r values
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        data.reshape(-1, 1))
    log_density = kde.score_samples(
        data_range.reshape(-1, 1))
    density = np.exp(log_density)

    # find peaks in density function
    d_density = np.diff(density)
    peak_idxs = [idx + 1 for idx, x in enumerate(zip(d_density[:-1], d_density[1:]))
                 if x[0] >= 0.0 and x[1] < 0.0]

    if len(peak_idxs) == 0:
        peak_idxs = [np.argmax(density)]

    if visualize:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(data_range, density, color="blue")
        plt.plot(data_range[:-1], d_density, color="red")
        for peak_idx in peak_idxs:
            plt.axvline(x=data_range[peak_idx], color="green")
        plt.grid(True)
        plt.show(block=False)

    return peak_idxs, [density[idx] for idx in peak_idxs]


def mbs(arrays):
    """find the approximate size of a list of numpy arrays in MiB"""
    total = 0.0
    for array in arrays:
        total += array.nbytes / 1048576.0
    return np.round(total, 3)
