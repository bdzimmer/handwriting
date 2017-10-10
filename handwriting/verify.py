# -*- coding: utf-8 -*-
"""

Interactively verify predictions from algorithms so they can be used as ground
truth for evaluation or training.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import os
import pickle
import sys

import cv2
import numpy as np

from handwriting import annotate, extract, findletters, findwords, findlines
from handwriting.prediction import Prediction


def verify_predictions(
        preds,
        select_func,
        annotate_func,
        review_mode=False):

    """given a list of predictions, allow the user to interactively confirm or
    fix unverified predictions.

    preds - list of Predictions
    select_func - method for selecting which unverified prediction to verify next.
             Takes a list of unverified preds,
            returns index or None if done.
    annotate_func - function to allow interactive confirmation / fixing; takes data
            and result and returns new result. If result is none, the prediction
            will be eliminated from the dataset.
    review_mode - if true, include verified predictions

    returns new list of updated predictions

    """

    # copy the list
    if review_mode:
        preds_verified = []
        preds = [x for x in preds]
    else:
        preds_verified = [x for x in preds if x.verified]
        preds = [x for x in preds if not x.verified]

    while True:
        idx = select_func(preds)
        if idx is None:
            break
        pred = preds.pop(idx)
        result = annotate_func(pred.data, pred.result)
        if result is not None:
            pred_verified = pred.copy(result=result, trust=1.0, verified=True)
            preds_verified.append(pred_verified)

    return preds + preds_verified


def main(argv):
    """main program"""

    input_filename = argv[1]
    verify_type = argv[2]

    # some prediction selector functions

    def build_select(n):
        """build a selector that selects n predictions before quitting"""
        count = [0]
        def select(xs):
            """do the selecting"""
            print("totaled unverified predictions remaining:", len(xs))
            print("predictions verified:", count[0], "/", n)
            if len(xs) == 0 or count[0] >= n:
                return None
            count[0] += 1
            # TODO: do this without a random choice
            # easier to annotate characters latter if in order
            return np.random.choice(range(len(xs)), 1)[0]
        return select

    def select_all(xs):
        """select all predictions before quitting"""
        if len(xs) == 0:
            return None
        return 0

    preds_filename = input_filename + "." + verify_type + ".pkl"

    if os.path.exists(preds_filename):
        print("loading predictions from file")
        with open(preds_filename, "rb") as preds_file:
            preds = pickle.load(preds_file)
        preds_loaded = True
    else:
        preds_loaded = False

    if verify_type == "lines":

        if not preds_loaded:
            print("generating initial line position predictions")
            image = cv2.imread(input_filename)
            lines = findlines.find(image)[1]
            preds = [Prediction(image, lines, 0.0, False)]

        print("Verify the positions of the lines.")
        print("left mouse button:  create a new line with two clicks")
        print("right mouse button: delete the nearest line")
        print("escape: done")
        print()
        preds_verified = verify_predictions(preds, select_all, annotate.annotate_lines)

    if verify_type == "wpos":

        # verify word positions

        if not preds_loaded:
            print("generating initial word position predictions from text lines")
            # px_above, px_below = 48, 32
            px_above, px_below = 72, 32

            if False:
                image = cv2.imread(input_filename)
                lines_filename = input_filename + ".lines.text"
                with open(lines_filename, "rb") as lines_file:
                    lines = pickle.load(lines_file)
            else:
                with open(input_filename + ".lines.pkl", "rb") as lines_file:
                    line_preds = pickle.load(lines_file)
                # for now, assume one prediction
                image = line_preds[0].data
                lines = line_preds[0].result

            line_images = [extract.extract_line_image(line, image, px_above, px_below)
                           for line in lines]
            find_word_positions = findwords.find_thresh
            preds = [Prediction(x, find_word_positions(x), 0.0, False)
                     for x in line_images]

        print("Verify the positions of the words.")
        print("left mouse button:  create a new word with two clicks")
        print("right mouse button: delete the nearest word")
        print("escape:             done")
        print()

        preds_verified = verify_predictions(
            preds, select_all, annotate.annotate_word_positions)

    elif verify_type == "lgap":

        # verify letter gaps

        if not preds_loaded:
            print("generating initial letter gaps from verified word positions")

            with open(input_filename + ".wpos.pkl", "rb") as wpos_file:
                word_positions = pickle.load(wpos_file)

            find_letter_positions = findletters.find_gaps_thresh_peaks

            def init_lgap_prediction(image, wpos):
                """helper"""
                word_image = image[:, wpos[0]:wpos[1]]
                lgaps = find_letter_positions(word_image)
                return Prediction(word_image, lgaps, 0.0, False)

            # TODO: filter verified predictions

            preds = [init_lgap_prediction(x.data, y)
                     for x in word_positions
                     for y in x.result]

        print("Verify the positions of gaps between letters.")
        print("left mouse button:  create a new gap")
        print("right mouse button: delete the nearest gap")
        print("escape:             done")
        print()

        preds_verified = verify_predictions(
            preds, build_select(50), annotate.annotate_letter_gaps)

    elif verify_type == "character":

        # verify character classification

        if not preds_loaded:
            print("generating initial character classifications")

            with open(input_filename + ".lgap.pkl", "rb") as lgap_file:
                letter_gaps = pickle.load(lgap_file)

            letter_gaps = [x for x in letter_gaps if x.verified]

            char_data = [(y, x.data)
                         for x in letter_gaps
                         for y in extract.extract_letters(x.data, x.result)]

            preds = [Prediction(x, "Z", 0.0, False) for x in char_data]

        print("Verify character classification.")
        print("key:    assign character")
        print("escape: done")
        print()

        preds_verified = verify_predictions(
            preds, build_select(0),
            lambda x, y: annotate.annotate_character(x[0], x[1], y))

    with open(preds_filename, "wb") as preds_file:
        pickle.dump(preds_verified, preds_file)


if __name__ == "__main__":
    main(sys.argv)
