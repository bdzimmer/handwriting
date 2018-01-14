# -*- coding: utf-8 -*-
"""

Interactively verify predictions from algorithms so they can be used as ground
truth for evaluation or training.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

# New process that uses mutable tree of Samples.


import os
import sys

import cv2
import numpy as np

from handwriting import analysisimage, annotate, driver
from handwriting import findletters, charclass, improc, util
from handwriting.prediction import Sample


def int_safe(obj, default=0):
    """safely convert something to an integer"""
    try:
        obj_int = int(obj)
    except ValueError:
        obj_int = default
    return obj_int


def _image_idx(x_val, widths, hgap):
    """find the index of an image given a position, image widths, and a
    horizontal gap size"""
    widths = [x + hgap for x in widths]
    widths_cumsum = np.cumsum([0] + widths)
    return np.where(x_val >= widths_cumsum)[0][-1]


def _mutate_set_verify_recursive(sample, verified):
    """recursively set a hierarchy of samples as verified or unverified"""
    sample.verified = verified
    if isinstance(sample.result, Sample):
        _mutate_set_verify_recursive(sample.result, verified)
    elif isinstance(sample.result, list):
        for samp in sample.result:
            _mutate_set_verify_recursive(samp, verified)


def _verification_status_recursive(sample, verified=0, total=0):
    """recursively determine how much of the sample has been verified"""
    total = total + 1
    if sample.verified:
        verified = verified + 1
    if isinstance(sample.result, Sample):
        verified, total = _verification_status_recursive(
            sample.result, verified, total)
    elif isinstance(sample.result, list):
        for samp in sample.result:
            verified, total = _verification_status_recursive(
                samp, verified, total)
    return verified, total



def _mutate_recalculate_list(
        list_update, new_items, compare_func, calc_func):
    """update samples in a list, recalculating items that have changed"""
    # list_update is a list of Samples
    # new_items are not samples; calc_func will take one of these and return
    # a Sample
    res = []
    for item in new_items:
        found = False
        for old_item in list_update:
            if compare_func(item, old_item.data):
                print(old_item.data)
                # y.verified = True
                res.append(old_item)
                found = True
                break
        if not found:
            print("recalculating", item)
            sample = calc_func(item)
            # sample.verified = True
            res.append(sample)
    print("done updating list")
    list_update[:] = res


def _mutate_verify_line_poss(image_sample, process_line_position):
    """verify positions of lines"""

    print("Verify the positions of the lines.")
    print("left mouse button:  create a new line with two clicks")
    print("right mouse button: delete the nearest line")
    print("escape: done")
    print()

    lines = [x.data for x in image_sample.result]

    lines_verified = annotate.annotate_lines(image_sample.data, lines)

    # update what's been modified in the hierarchy
    calc_func = lambda x: process_line_position(x, image_sample.data)

    _mutate_recalculate_list(
        image_sample.result, lines_verified, np.allclose, calc_func)

    image_sample.verified = True
    for samp in image_sample.result: # verify line position samples
        samp.verified = True

    cv2.destroyWindow("line analysis image")


def _mutate_verify_multi(
        line_image_sample,
        process_word_position,
        process_char_position,
        new_char_annotation_mode):

    """open different annotation options depending on click location
    in line analysis image"""

    def draw():
        """refresh the view"""
        lai = analysisimage.LineAnalysisImage(line_image_sample)
        cv2.imshow("line analysis image", lai.image)

    def on_mouse(event, mouse_x, mouse_y, flags, params):
        """helper"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(mouse_x, mouse_y, "left")
            lai = analysisimage.LineAnalysisImage(line_image_sample)

            if mouse_y >= lai.line_y_start and mouse_y < lai.line_y_end:
                print("line")
                print("Verify the positions of the words.")
                print("left mouse button:  create a new word with two clicks")
                print("right mouse button: delete the nearest word")
                print("escape:             done")
                print()

                word_positions = [x.data for x in line_image_sample.result]
                words_verified = annotate.annotate_word_positions(
                    line_image_sample.data, word_positions)
                calc_func = lambda x: process_word_position(x, line_image_sample.data)
                _mutate_recalculate_list(
                    line_image_sample.result, words_verified, np.allclose, calc_func)
                line_image_sample.verified = True
                for samp in line_image_sample.result: # verify word position samples
                    samp.verified = True
                draw()

            elif mouse_y >= lai.words_y_start and mouse_y < lai.words_y_end:
                print("words")

                # which word are we modifying?
                word_positions = line_image_sample.result
                idx = _image_idx(
                    mouse_x - lai.all_words_im_x,
                    [word_pos.data[1] - word_pos.data[0] for word_pos in word_positions],
                    analysisimage.HGAP_LARGE)

                # TODO: work with word image sample instead
                word_position_sample = word_positions[idx]

                char_positions = [x.data for x in word_position_sample.result.result]
                print("char positions:", char_positions)

                if new_char_annotation_mode:
                    print("Verify the positions of the characters.")
                    print("left mouse button:  create a new character with two clicks")
                    print("right mouse button: delete the nearest word")
                    print("escape:             done")
                    print()
                    char_positions_verified = annotate.annotate_word_positions(
                        word_position_sample.result.data,
                        char_positions)
                else:
                    print("Verify the positions of gaps between letters.")
                    print("left mouse button:  create a new gap")
                    print("right mouse button: delete the nearest gap")
                    print("escape:             done")
                    print()
                    char_gaps = findletters.positions_to_gaps(char_positions)
                    char_gaps_verified = annotate.annotate_letter_gaps(
                        word_position_sample.result.data,
                        char_gaps)
                    char_positions_verified = findletters.gaps_to_positions(char_gaps_verified)

                print("char positions verified:", char_positions_verified)
                calc_func = lambda x: process_char_position(x, word_position_sample.result.data)
                _mutate_recalculate_list(
                    word_position_sample.result.result, char_positions_verified,
                    np.allclose, calc_func)
                word_position_sample.result.verified = True # verify word image sample
                for samp in word_position_sample.result.result: # verify char position samples
                    samp.verified = True
                draw()

            elif mouse_y >= lai.char_ims_y_start and mouse_y < lai.char_ims_y_end:
                # verify character labels by word
                print("char ims")

                # which word are we modifying?
                word_positions = line_image_sample.result
                idx = _image_idx(
                    mouse_x - lai.all_char_ims_im_x,
                    [np.sum([char_pos.data[1] - char_pos.data[0] + analysisimage.HGAP_SMALL
                             for char_pos in word_pos.result.result]) - analysisimage.HGAP_SMALL
                     for word_pos in word_positions],
                    analysisimage.HGAP_LARGE)

                patch_width = 96
                patch_height = 96
                pad = lambda x: improc.pad_image(x, patch_width, patch_height)
                # TODO: most of this logic is to deal with the charclass interface
                def pad_preds(preds):
                    """helper"""
                    return [p.copy(data=(pad(p.data), None, p.data)) for p in preds]
                def unpad_preds(preds):
                    """helper"""
                    return [p.copy(data=(p.data[2], p.data[1])) for p in preds]

                while idx < len(word_positions):
                    char_img_samples = [char_pos.result
                                        for char_pos in word_positions[idx].result.result]
                    chars_working, chars_done = charclass.label_chars(pad_preds(char_img_samples))
                    # this is a bit of a hack, but it works well for now.
                    print(len(chars_working), len(chars_done))
                    if len(chars_done) == 0:
                        break
                    char_img_samples_verified = unpad_preds(chars_working) + unpad_preds(chars_done)
                    for org_sample, new_sample in zip(char_img_samples, char_img_samples_verified):
                        org_sample.result = new_sample.result
                        org_sample.verified = new_sample.verified
                    draw()
                    idx = idx + 1

            elif mouse_y >= lai.chars_y_start and mouse_y < lai.chars_y_end:
                print("chars")

            cv2.waitKey(1)
        if event == cv2.EVENT_RBUTTONDOWN:
            print(mouse_x, mouse_y, "right")
            cv2.waitKey(1)

    cv2.namedWindow("line analysis image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("line analysis image", on_mouse, 0)
    draw()

    while True:
        key = cv2.waitKey(0)
        if key == 27:
            break
        if key == 13:
            _mutate_set_verify_recursive(line_image_sample, True)
        if key == 8:
            _mutate_set_verify_recursive(line_image_sample, False)
        draw()

    cv2.destroyWindow("line analysis image")


def main(argv):
    """main program"""

    if len(argv) < 3:
        print("Usage: verify input_file <line | multi | view>")
        sys.exit()

    input_filename = argv[1]
    verify_type = argv[2]

    new_char_annotation_mode = False

    # filename has a number version suffix
    sample_filename = input_filename + ".sample.pkl"
    sample_dirname, sample_basename = os.path.split(sample_filename)

    possible_files = [x for x in os.listdir(sample_dirname)
                      if x.startswith(sample_basename)]
    versions = [int_safe(x.split(".")[-1]) for x in possible_files]
    latest_idx = np.argmax(versions)

    latest_version = versions[latest_idx]
    latest_filename = possible_files[latest_idx]

    sample_filename_full = os.path.join(sample_dirname, latest_filename)
    print("loading sample file:", sample_filename_full)
    image_sample = util.load(sample_filename_full)
    # with open(sample_filename_full, "rb") as sample_file:
    #     image_sample = pickle.load(sample_file)

    status = _verification_status_recursive(image_sample)
    print(
        status[0], "/", status[1], "samples verified", "-",
        np.round(status[0] / status[1] * 100, 2), "%")

    (process_image,
     process_line_position,
     process_word_position,
     process_char_position) = driver.current_best_process()

    if verify_type == "line":
        _mutate_verify_line_poss(image_sample, process_line_position)
    elif verify_type == "view":
        for line_pos in image_sample.result:
            img = analysisimage.LineAnalysisImage(line_pos.result).image
            cv2.namedWindow("line analysis", cv2.WINDOW_NORMAL)
            cv2.imshow("line analysis", img)
            cv2.waitKey()
    else:
        for line_pos in image_sample.result:
            _mutate_verify_multi(
                line_pos.result,
                process_word_position, process_char_position,
                new_char_annotation_mode)

    if verify_type != "view":
        status = _verification_status_recursive(image_sample)
        print(
            status[0], "/", status[1], "samples verified", "-",
            np.round(status[0] / status[1] * 100, 2), "%")

        sample_filename_full = sample_filename + "." + str(latest_version + 1)
        print("writing sample file:", sample_filename_full)
        util.save(image_sample, sample_filename_full)
        # with open(sample_filename_full, "wb") as sample_file:
        #     pickle.dump(image_sample, sample_file)


if __name__ == "__main__":
    main(sys.argv)
