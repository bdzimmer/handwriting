# -*- coding: utf-8 -*-
"""

Integration test for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


from functools import partial
import string
import unittest

import numpy as np

from handwriting import charclassml as cml, analysisimage, charclass, ml
from handwriting.prediction import Sample

VISUALIZE = False


class TestsCharClassML(unittest.TestCase):

    """Integration test for character classification."""

    def test_process(self):
        """test the full classification process"""

        print("generating synthetic data...", end="")

        data_single, labels_single = _generate_samples()

        # perturb to create 10 examples of each label
        balance_factor = 10
        data, labels = ml.balance(
            data_single,
            labels_single,
            balance_factor,
            partial(
                ml.transform_random,
                trans_size=2.0,
                rot_size=0.3,
                scale_size=0.1))

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
            data_train, labels_train, support_ratio_max=0.97)

        (classify_char_image,
         prep_image, feat_extractor, feat_selector,
         classifier, model) = res

        # feats_test = feat_selector([feat_extractor(x) for x in data_test])
        # score = ml.score_auc(model, feats_test, labels_test)
        # print("score on test dataset", score)
        # self.assertGreater(score, 0.5)

        print("done")

        labels_test_pred = classify_char_image(data_test)

        if VISUALIZE:
            chars_confirmed = []
            chars_redo = []
            # show results
            for cur_label, group in ml.group_by_label(data_test, labels_test_pred):
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
