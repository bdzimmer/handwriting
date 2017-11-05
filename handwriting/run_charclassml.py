# -*- coding: utf-8 -*-
"""

Train a character classifier.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import random

import numpy as np
import sklearn

from handwriting import charclassml as cml, util, charclass
from handwriting.prediction import Sample

VISUALIZE = False


def _load_samples(filenames):
    """load verified character label samples from multiple files"""

    # load multiple files
    line_poss = [y for x in filenames for y in util.load(x).result]

    # get word images and verified character positions
    word_ims = [word_pos.result
                for line_pos in line_poss
                for word_pos in line_pos.result.result
                if word_pos.result.verified]

    # only keep if word contains other letters besides "`"
    # and all of the character labels have been verified
    char_poss = [char_pos
                 for word_im in word_ims
                 for char_pos in word_im.result
                 if char_pos.result.verified]

    return [x.result.data for x in char_poss], [x.result.result for x in char_poss]


def main():
    """main program"""

    np.random.seed(0)
    random.seed(0)

    model_filename = "models/classify_characters.pkl"
    min_label_examples = 1
    remove_labels = ["\"", "!", "/", "~"]
    balance_factor = 50
    support_ratio_max = 0.9

    sample_filenames = ["data/20170929_" + str(idx) + ".png.sample.pkl.1"
                        for idx in range(1, 6)]
    train_filenames = sample_filenames[0:4]
    test_filenames = sample_filenames[4:5]

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = _load_samples(train_filenames)

    # eliminate groups from training and test
    # where we have less than a certain number of samples or they aren't
    # characters that we currently want to train on
    train_gr = dict(cml.group_by_label(
        data_train_unbalanced, labels_train_unbalanced))
    keep_labels = sorted(
        [x for x, y in train_gr.items()
         if len(y) >= min_label_examples and x not in remove_labels])
    print("keep labels:", keep_labels)

    train_grf = {x: y for x, y in train_gr.items() if x in keep_labels}
    data_train_unbalanced, labels_train_unbalanced = zip(*[
        (y, x[0]) for x in train_grf.items() for y in x[1]])

    print(
        "training group sizes before balancing:",
        [(x[0], len(x[1]))
         for x in cml.group_by_label(
             data_train_unbalanced, labels_train_unbalanced)])

    # balance classes in training set
    data_train, labels_train = cml.balance(
        data_train_unbalanced, labels_train_unbalanced,
        balance_factor, cml.transform_random)

    # load test set
    data_test, labels_test = _load_samples(test_filenames)

    test_gr = dict(cml.group_by_label(data_test, labels_test))
    test_grf = {x: y for x, y in test_gr.items() if x in keep_labels}
    data_test, labels_test = zip(*[
        (y, x[0]) for x in test_grf.items() for y in x[1]])

    print("done")

    print("training size:", len(data_train))
    print("test size:", len(data_test))

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in cml.group_by_label(data_train, labels_train)])

    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in cml.group_by_label(data_test, labels_test)])

    print("training model...")

    proc = cml.build_current_best_process(
        data_train, labels_train, support_ratio_max)

    (classify_char_image,
     prep_image, feat_extractor, feat_selector,
     classifier, classifier_score) = proc

    print("done")

    # feats_test = feat_selector([feat_extractor(x) for x in data_test])
    # print("score on test dataset:", classify_char_image(feats_test, labels_test))
    labels_test_pred = classify_char_image(data_test)
    print("score on test dataset:", sklearn.metrics.accuracy_score(labels_test, labels_test_pred))

    print("confusion matrix:")
    confusion_mat = sklearn.metrics.confusion_matrix(
        labels_test, labels_test_pred, keep_labels)
    print(confusion_mat)
    np.savetxt(
        model_filename + ".confusion.tsv",
        confusion_mat,
        fmt="%d",
        delimiter="\t",
        header="\t".join(keep_labels))

    # TODO: visualize ROC curves

    util.save_dill(proc, model_filename)

    if VISUALIZE:
        labels_test_pred = classify_char_image(data_test)
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


if __name__ == "__main__":
    main()
