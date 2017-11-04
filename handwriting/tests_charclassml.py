# -*- coding: utf-8 -*-
"""

Unit tests for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import string
import unittest

import cv2
import numpy as np

from handwriting import charclassml as cml, analysisimage, charclass
from handwriting.prediction import Sample

VISUALIZE = False


class TestsCharClassML(unittest.TestCase):

    """Unit tests for character classification."""

    def test_balance(self):
        """test dataset balancing functionality"""

        data, labels = zip(*([(0.0, "a")] * 4 + [(1.0, "b")] * 1))

        balanced_data, balanced_labels = cml.balance(
            data, labels, 0.5, lambda x: x)
        balanced_grouped = dict(cml.group_by_label(
            balanced_data, balanced_labels))
        for label, group in balanced_grouped.items():
            self.assertEqual(len(group), 2)

        balanced_data, balanced_labels = cml.balance(
            data, labels, 8, lambda x: x)
        balanced_grouped = dict(cml.group_by_label(
            balanced_data, balanced_labels))
        for label, group in balanced_grouped.items():
            self.assertEqual(len(group), 8)


    def test_pad_image(self):
        """test function to pad an image"""

        # image is smaller in both dimensions
        image = np.ones((8, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = cml.pad_image(image, 16, 16)
        self.assertEqual(image_padded.shape, (16, 16, 3))

        if VISUALIZE:
            image_both = np.zeros((16, 32, 3), dtype=np.uint8)
            image_both[0:image.shape[0], 0:image.shape[1], :] = image
            image_both[0:image_padded.shape[0], 16:(image_padded.shape[1] + 16)] = image_padded
            cv2.namedWindow("padding", cv2.WINDOW_NORMAL)
            cv2.imshow("padding", image_both)
            cv2.waitKey()

        # image is larger in both dimensions
        image = np.ones((16, 16, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = cml.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is larger in x dimension
        image = np.ones((8, 16, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = cml.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is larger in y dimension
        image = np.ones((16, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = cml.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))

        # image is same in both dimensions
        image = np.ones((8, 8, 3), dtype=np.uint8) * (0, 0, 255)
        image_padded = cml.pad_image(image, 8, 8)
        self.assertEqual(image_padded.shape, (8, 8, 3))


    def test_process(self):
        """test the full classification process"""

        print("generating synthetic data...", end="")

        data_single, labels_single = _generate_samples()

        # perturb to create 40 examples of each label
        balance_factor = 10
        data, labels = cml.balance(
            data_single, labels_single,
            balance_factor, cml.transform_random)

        # split 75% / 25%  into training and test set
        train_count = int(balance_factor * 0.75)
        train_idxs_bool_single = np.hstack((
            np.repeat(True, train_count),
            np.repeat(False, balance_factor - train_count)))
        train_idxs_bool = np.tile(
            train_idxs_bool_single, int(len(data) / balance_factor))
        test_idxs_bool = np.logical_not(train_idxs_bool)
        train_idxs = np.where(train_idxs_bool)[0]
        test_idxs = np.where(test_idxs_bool)[0]

        data_train = [data[i] for i in train_idxs]
        labels_train = [labels[i] for i in train_idxs]
        data_test = [data[i] for i in test_idxs]
        labels_test = [labels[i] for i in test_idxs]

        print("done")

        print("training size:", len(data_train))
        print("test size:", len(data_test))

        print("training model")

        res = cml.build_current_best_process(
            data_train, labels_train, support_ratio_max=0.95)

        (classify_char_image,
         prep_image, feat_extractor, feat_selector,
         classifier, classifier_score) = res

        feats_test = feat_selector([feat_extractor(x) for x in data_test])
        score = classifier_score(feats_test, labels_test)
        print("score on test dataset", score)

        self.assertGreater(score, 0.5)

        print("done")

        labels_test_pred = classify_char_image(data_test)

        if VISUALIZE:
            chars_confirmed = []
            chars_redo = []
            # show results
            for cur_label, group in cml.group_by_label(data_test, labels_test_pred):
                print(cur_label)
                group_prepped = [(prep_image(x), None) for x in group]
                group_pred = [Sample(x, cur_label, 0.0, False) for x in group_prepped]
                chars_working, chars_done = charclass.label_chars(group_pred)
                chars_confirmed += chars_working
                chars_redo += chars_done


def _generate_samples():
    """generate a single sample of each character we want to test classifying"""

    characters = string.ascii_letters + string.digits + "(),:;/'\"* "

    labels = [x for x in characters]
    data = [analysisimage.char_image_var_width(x, 104)
            for x in labels]
    return data, labels
