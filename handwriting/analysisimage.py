# -*- coding: utf-8 -*-
"""
Analysis image class and related functions.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import cv2
import numpy as np


HGAP_SMALL = 4
HGAP_LARGE = 32
VGAP = 16


class LineAnalysisImage(object):
    """Construct a line analysis image, keeping track of where things are
    drawn on the image so we can use clicks for interaction"""

    def __init__(self, line_im_sample):
        """constructor"""

        self.line_im_sample = line_im_sample
        self.line_height = line_im_sample.data.shape[0]

        self.line_y_start = 0
        self.line_y_end = self.line_y_start + self.line_height

        self.words_y_start = self.line_y_end + VGAP
        self.words_y_end = self.words_y_start + self.line_height

        self.char_ims_y_start = self.words_y_end + VGAP
        self.char_ims_y_end = self.char_ims_y_start + self.line_height

        self.chars_y_start = self.char_ims_y_end + VGAP
        self.chars_y_end = self.chars_y_start + self.line_height

        word_positions = [x for x in self.line_im_sample.result]
        char_positions_by_word = [y for x in self.line_im_sample.result
                                  for y in x.result.result]
        (self.image,
         self.line_im_x,
         self.all_words_im_x,
         self.all_char_ims_im_x,
         self.all_char_ims_x) = line_analysis_image(
             self.line_im_sample,
             word_positions,
             char_positions_by_word, HGAP_SMALL, HGAP_LARGE, VGAP)


def line_analysis_image(
        line_im,
        word_positions,
        char_positions_by_word,
        hgap_small, hgap_large, vgap):
    """illustrate how an entire line was analyzed"""

    # line_im                - line image sample
    # word_positions         - list of word position samples
    # char_positions_by_word - for each word, a list of char position samples

    line_height = line_im.data.shape[0]

    hgap_small_im = np.zeros((line_height, hgap_small, 3), dtype=np.uint8)
    hgap_large_im = np.zeros((line_height, hgap_large, 3), dtype=np.uint8)

    def combine_ims(ims, sep_im):
        """combine images horizontally with a separating image in between"""
        if len(ims) == 0:
            return np.zeros((line_height, 0, 3), dtype=np.uint8)
        else:
            return np.hstack([y for x in ims for y in [x, sep_im]][:-1])

    def im_ver(img, verified):
        """helper"""
        res = np.copy(img)
        # res = 255 - im
        if not verified:
            res[-8:, :, :] = (0, 0, 255)
        return res

    # extract data fields from samples

    # both line image and word position must be verified
    word_ims = [im_ver(x.result.data, line_im.verified and x.verified)
                for x in word_positions]
    # both word image and char position must be verified
    char_ims_by_word = [[im_ver(y.result.data, x.result.verified and y.verified)
                         for y in x.result.result]
                        for x in word_positions]
    # char image must be verified
    chars_by_word = [[(y.result.result, y.result.verified) for y in x.result.result]
                     for x in word_positions]

    # build the image

    all_words_im = combine_ims(word_ims, hgap_large_im)

    all_char_ims_im = combine_ims(
        [combine_ims(x, hgap_small_im)
         for x in char_ims_by_word], hgap_large_im)

    def char_image(ch):
        """draw character image"""
        char_im = np.zeros((line_height, 20, 3), dtype=np.uint8)
        cv2.putText(
            char_im, ch, (1, int(line_height / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        return char_im

    all_chars_im = combine_ims(
        [combine_ims([im_ver(char_image(y[0]), y[1]) for y in x], hgap_small_im)
         for x in chars_by_word], hgap_large_im)

    max_width = np.max([
        line_im.data.shape[1],
        all_words_im.shape[1],
        all_char_ims_im.shape[1],
        all_chars_im.shape[1]])

    def pad(img):
        """helper"""
        padded_im = np.zeros((line_height, max_width, 3), dtype=np.uint8)
        start_x = int(0.5 * (max_width - img.shape[1]))
        padded_im[:, start_x:(start_x + img.shape[1]), :] = img
        return padded_im, start_x

    vgap_im = np.zeros((vgap, max_width, 3), dtype=np.uint8)

    line_im_pad, line_im_x = pad(line_im.data)
    all_words_im_pad, all_words_im_x = pad(all_words_im)
    all_char_ims_im_pad, all_char_ims_im_x = pad(all_char_ims_im)
    all_chars_im_pad, all_char_ims_x = pad(all_chars_im)

    res = np.vstack((
        line_im_pad,
        vgap_im,
        all_words_im_pad,
        vgap_im,
        all_char_ims_im_pad,
        vgap_im,
        all_chars_im_pad))

    return res, line_im_x, all_words_im_x, all_char_ims_im_x, all_char_ims_x
