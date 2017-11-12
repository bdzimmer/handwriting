# -*- coding: utf-8 -*-
"""

Train a classifier to classify vertical slices of a word image as within a
a letter or between letters.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import random
import sys

import numpy as np
import sklearn

from handwriting import util, charclass, ml
from handwriting import findletters, findwords
from handwriting.prediction import Sample

VISUALIZE = False


def build_classification_process(data_train, labels_train):

    """build a classification process"""

    # A lot of this is reused from the character classification process

    pad_image_96 = lambda x: ml.pad_image(x, 96, 96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        # start_row = 16
        start_row = 32
        image = image[start_row:, :]
        # return ml._align(ml._filter_cc(pad_image_96(image)))
        # return pad_image_96(image)
        return ml._align(pad_image_96(image))

    def feat_extractor(image):
        """convert image to feature vector"""
        img_p = prep_image(image)
        img_g = ml.grayscale(img_p)
        # return ml._max_pool_multi(img_g, [3, 4])
        # return ml._downsample_4(img_g)
        # return ml._downsample_multi(img_g, [0.125])# [0.5, 0.25, 0.125])
        # grad_0, grad_1 = np.gradient(img_g)
        grad_0, grad_1 = np.gradient(ml._max_pool(img_g))
        grad_0 = np.abs(grad_0)
        grad_1 = np.abs(grad_1)
        return np.hstack((np.ravel(grad_0), np.ravel(grad_1)))
        # return np.hstack((
        #     ml._max_pool_multi(grad_0, [2]),
        #     ml._max_pool_multi(grad_1, [2])))

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = ml.build_feat_selection_pca(feats_train, 0.94) # 0.94
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = ml.train_classifier(
        fit_model=partial(
            ml.build_svc_fit,
            support_ratio_max=0.8),
        score_func=ml.score_accuracy,
        n_splits=4,
        feats=feats_train,
        labels=labels_train,
        c=np.logspace(1, 2, 4),
        gamma=np.logspace(-2, 0, 7))

    classify_char_image = ml.build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)


def _load_words(filenames):
    """load verified word image samples from multiple files"""

    # load multiple files
    line_poss = [y for x in filenames for y in util.load(x).result]

    # get word images and verified character positions
    word_ims = [word_pos.result
                for line_pos in line_poss
                for word_pos in line_pos.result.result
                if word_pos.result.verified]

    # only keep if word contains other letters besides "`"
    # and all of the character labels have been verified
    word_ims = [word_im for word_im in word_ims
                if np.sum([char_pos.result.result != "`"
                           for char_pos in word_im.result]) > 0
                and np.sum([char_pos.result.verified
                            for char_pos in word_im.result]) == len(word_im.result)]

    # print the words
    for word_im in word_ims:
        print("".join([char_pos.result.result
                       for char_pos in word_im.result]), end=" ")

    return word_ims


def _load_samples(filenames, half_width):
    """load word image slices from multiple files"""

    # load multiple files
    line_poss = [y for x in filenames for y in util.load(x).result]

    # get word images and verified character positions
    word_ims = [word_pos.result
                for line_pos in line_poss
                for word_pos in line_pos.result.result
                if word_pos.result.verified]

    # helper functions for extracting images
    extract_char = lambda cpos, im: im[:, np.maximum(cpos[0], 0):cpos[1]]
    extract_char_half_width = lambda x, im: extract_char((x - half_width, x + half_width), im)

    def keep_valid(ims):
        """keep images with width greater than 0"""
        return [x for x in ims if x.shape[1] > 0]

    def half(x, y):
        """point halfway between two points"""
        return int((x + y) * 0.5)

    def extract(extract_func):
        """extract images from valid char positions using extract_func"""
        res = [(extract_func(char_pos.data, word_im.data), char_pos.result.result, char_pos.data, word_im.data)
                for word_im in word_ims
                for char_pos in word_im.result
                if char_pos.result.result != "`"
                and char_pos.result.verified
                and (char_pos.data[1] - char_pos.data[0]) > 1]
        return res

    # extract images from the ends of all positions in each word
    char_start_ims = (
        extract(lambda x, y: extract_char_half_width(x[1], y)) +
        extract(lambda x, y: extract_char_half_width(x[1] + 1, y)) +
        extract(lambda x, y: extract_char_half_width(x[1] - 1, y)))

    # extract images from half, one fourth, and three fourths of the way
    # between starts and ends of each position
    char_middle_ims = (
        extract(lambda x, y: extract_char_half_width(
            half(x[0], x[1]), y)) +
        extract(lambda x, y: extract_char_half_width(
            half(x[0], half(x[0], x[1])), y)) +
        extract(lambda x, y: extract_char_half_width(
            half(half(x[0], x[1]), x[1]), y)))

    # import cv2
    # for im in char_start_ims:
    #     print(im[0].shape, im[1], im[2], half(im[2][0], im[2][1]), im[3].shape)
    #     cv2.imshow("test", im[0])
    #     cv2.waitKey()

    data_with_src = char_start_ims + char_middle_ims
    labels = [True] * len(char_start_ims) + [False] * len(char_middle_ims)

    combined_labels = [(x, y[1]) for x, y in zip(labels, data_with_src)]

    data = [x[0] for x in data_with_src]
    return data, combined_labels



def main(argv):
    """main program"""

    if len(argv) < 2:
        mode = "tune"
    else:
        mode = argv[1]

    np.random.seed(0)
    random.seed(0)

    model_filename = "models/classify_charpos.pkl"
    half_width = 8
    balance_factor = 15 # 15 # 500

    sample_filenames = ["data/20170929_" + str(idx) + ".png.sample.pkl.1"
                        for idx in range(1, 6)]
    train_filenames = sample_filenames[0:4]
    test_filenames = sample_filenames[4:5]

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = _load_samples(
        train_filenames, half_width)

    print(
        "training group sizes before balancing:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(
             data_train_unbalanced, labels_train_unbalanced)])

    # balance classes in training set
    data_train, labels_train = ml.balance(
        data_train_unbalanced, labels_train_unbalanced,
        balance_factor, ml.transform_random)

    # load test set
    data_test, labels_test = _load_samples(test_filenames, half_width)

    test_gr = dict(ml.group_by_label(data_test, labels_test))
    test_grf = test_gr
    data_test, labels_test = zip(*[
        (y, x[0]) for x in test_grf.items() for y in x[1]])

    print("done")

    print("training size:", len(data_train))
    print("test size:", len(data_test))

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_train, labels_train)])
    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_test, labels_test)])

    print("discarding letter information from labels")

    labels_train = [x[0] for x in labels_train]
    labels_test = [x[0] for x in labels_test]

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_train, labels_train)])
    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_test, labels_test)])

    if mode == "train":

        print("training model...")

        proc = build_classification_process(
            data_train, labels_train)

        (classify_char_pos,
         prep_image, feat_extractor, feat_selector,
         classifier, classifier_score) = proc

        print("done")

        # feats_test = feat_selector([feat_extractor(x) for x in data_test])
        # print("score on test dataset:", classifier_score(feats_test, labels_test))
        labels_test_pred = classify_char_pos(data_test)
        print("score on test dataset:", sklearn.metrics.accuracy_score(
            labels_test, labels_test_pred))

        print("confusion matrix:")
        print(sklearn.metrics.confusion_matrix(
            labels_test, labels_test_pred, [True, False]))

        # TODO: visualize ROC curve

        util.save_dill(proc, model_filename)

        if VISUALIZE:
            labels_test_pred = classify_char_pos(data_test)
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

    if mode == "tune":

        # test different position finding methods using a distance function
        # on each word

        extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]

        print("loading test words...", end="", flush=True)
        word_ims_test = _load_words(test_filenames)
        print("done")

        def distance_test(find_char_poss_func):
            """helper"""
            distances = []
            for word_im in word_ims_test:
                positions = findletters.position_list_distance(
                    [x.data for x in word_im.result],
                    find_char_poss_func(word_im.data))
                distances.append(positions)
                print(".", end="", flush=True)
            # TODO: should I be aggregating this differently?
            return np.sum([y for x in distances for y in x])

        score = distance_test(findletters.find_thresh_peaks)
        print("position distance score - peaks:", score)

        find_comp_peaks = lambda word_im: findletters.find_combine(
            word_im, extract_char,
            lambda x: findwords.find_conc_comp(x, merge=False),
            findletters.find_thresh_peaks)
        score = distance_test(find_comp_peaks)
        print("position distance score - comps + peaks:", score)

        classify_char_pos = util.load_dill(model_filename)[0]
        find_classify = lambda word_im: findletters.find_classify(
            word_im, half_width, extract_char, classify_char_pos)
        score = distance_test(find_classify)
        print("position distance score - ML:", score)

        find_comp_peaks = lambda word_im: findletters.find_combine(
            word_im, extract_char,
            lambda x: findwords.find_conc_comp(x, merge=False),
            find_classify)
        score = distance_test(find_comp_peaks)
        print("position distance score - comps + ML:", score)


if __name__ == "__main__":
    main(sys.argv)
