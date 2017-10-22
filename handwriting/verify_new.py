# -*- coding: utf-8 -*-
"""

Interactively verify predictions from algorithms so they can be used as ground
truth for evaluation or training.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

# New process that uses mutable tree of Samples.


import os
import pickle
import sys

import cv2
import numpy as np

from handwriting import driver
from handwriting import annotate
from handwriting.driver import Sample


def int_safe(obj, default=0):
    """safely convert something to an integer"""
    try:
        obj_int = int(obj)
    except ValueError:
        obj_int = default
    return obj_int


def main(argv):
    """main program"""

    input_filename = argv[1]
    # verify_type = argv[2]

    # filename has a number version suffix
    sample_filename = input_filename + ".sample.pkl"
    sample_dirname, sample_basename = os.path.split(sample_filename)

    possible_files = [x for x in os.listdir(sample_dirname)
                      if x.startswith(sample_basename)]
    versions = [int_safe(x.split(".")[-1]) for x in possible_files]
    latest_idx = np.argmax(versions)

    latest_version = versions[latest_idx]
    latest_filename = possible_files[latest_idx]

    print("loading samples from file")
    with open(os.path.join(sample_dirname, latest_filename), "rb") as sample_file:
        image_sample = pickle.load(sample_file)

    (process_image,
     process_line,
     process_line_position,
     process_word,
     process_char) = driver.current_process()

    def _mutate_recalculate_list(
            list_update, new_items, compare_func, calc_func):
        """update samples in a list, recalculating items that have changed"""
        res = []
        for x in new_items:
            found = False
            for y in list_update:
                if compare_func(x, y.data):
                    print(y.data)
                    res.append(y)
                    found = True
                    break
            if not found:
                print("recalculating", x)
                res.append(calc_func(x))
        list_update[:] = res

    def _mutate_verify_line_poss(image_sample):

        print("Verify the positions of the lines.")
        print("left mouse button:  create a new line with two clicks")
        print("right mouse button: delete the nearest line")
        print("escape: done")
        print()

        lines = [x.data for x in image_sample.result]

        # get new line annotations and sort by Z
        # TODO: move Z sorting into annotate function
        lines_verified = annotate.annotate_lines(image_sample.data, lines)
        lines_verified = sorted(lines_verified, key=lambda x: 0.5 * (x[1] + x[3]))

        # update what's been modified in the hierarchy
        compare_func = lambda x, y: np.allclose(x, y)
        calc_func = lambda x: process_line_position(x, image_sample.data)

        _mutate_recalculate_list(
            image_sample.result, lines_verified, compare_func, calc_func)

    # TODO: other helper functions / options for verification and annotation

    _mutate_verify_line_poss(image_sample)

    for line_pos in image_sample.result:
        im = driver.line_sample_analysis_image(line_pos.result)
        cv2.namedWindow("line analysis", cv2.WINDOW_NORMAL)
        cv2.imshow("line analysis", im)
        cv2.waitKey()

    with open(sample_filename + "." + str(latest_version + 1), "wb") as sample_file:
        pickle.dump(image_sample, sample_file)


if __name__ == "__main__":
    main(sys.argv)
