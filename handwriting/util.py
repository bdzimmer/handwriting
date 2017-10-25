"""

General utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import dill
import numpy as np
from sklearn.neighbors import KernelDensity

dill.settings["recurse"] = True

VISUALIZE = False


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

    """combine up to 256 images into a larger image"""

    # TODO: get rid of default values for width and height
    patch_size = bmps[0].shape[0]
    res = np.zeros((height * patch_size, width * patch_size, 3), dtype=np.uint8)

    for idx in range(min(len(bmps), width * height)):
        col = idx % width * patch_size
        row = int(idx / width) * patch_size
        res[row:(row + patch_size), col:(col + patch_size)] = bmps[idx]

    return res


def find_peak_idxs(data, data_range, bandwidth):

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

    if VISUALIZE:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(data_range, density, color="blue")
        plt.plot(data_range[:-1], d_density, color="red")
        for peak_idx in peak_idxs:
            plt.axvline(x=data_range[peak_idx], color="green")
        plt.grid(True)
        plt.show(block=False)

    return peak_idxs, [density[idx] for idx in peak_idxs]
