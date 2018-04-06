# -*- coding: utf-8 -*-
"""

Train a classifier to classify vertical slices of a word image as within a
a letter or between letters.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

from functools import partial
import gc
import random
import sys

import attr
import cv2
import numpy as np
import sklearn
import torch

from handwriting import util, charclass, func, improc
from handwriting import ml, imml
from handwriting import findletters, findwords
from handwriting import data, config as cf
from handwriting.func import pipe
from handwriting.prediction import Sample

VISUALIZE = True
VERBOSE = False

MODE_TRAIN = "train"
MODE_TUNE = "tune"

IGNORE_CHARS = []


@attr.s
class Config:
    half_width = attr.ib()
    pad_height = attr.ib()
    start_row = attr.ib()
    offset = attr.ib()
    do_balance = attr.ib()
    balance_factor = attr.ib()
    trans_x_size = attr.ib()
    trans_y_size = attr.ib()
    rot_size = attr.ib()
    scale_size = attr.ib()
    batch_size = attr.ib()
    max_epochs = attr.ib()
    train_idxs = attr.ib()
    test_idxs = attr.ib()


CONFIG_DEFAULT = Config(
    half_width=16,
    pad_height=96,
    start_row=0,
    offset=0,
    do_balance=False,
    balance_factor=1024,
    trans_x_size=0.0,
    trans_y_size=0.0,
    rot_size=0.0,
    scale_size=0.0,
    batch_size=16,
    max_epochs=16,
    train_idxs=list(range(5, 15)),
    test_idxs=[15]
)


def build_distance_test(word_ims_test, char_poss_test):
    """create a distance test function to measure jaccard distance
    for various methods on test data"""

    def distance_test(find_char_poss_func, visualize=False):
        """helper"""
        jaccard_distance = lambda x, y: 1.0 - findletters.jaccard_index(x, y)
        distances = []

        ims = []

        for word_im, positions_true in zip(word_ims_test, char_poss_test):
            # positions_true = [x.data for x in word_im.result]
            positions_pred = find_char_poss_func(word_im)
            cur_distances, idxs = findletters.position_list_distance(
                positions_true, positions_pred,
                jaccard_distance)

            if visualize:
                # disp_im = _visualize_position_predictions(
                #     word_im.data,
                #     positions_true,
                #     positions_pred,
                #     idxs)

                disp_im = _visualize_position_predictions_stacked(
                    word_im,
                    positions_true,
                    positions_pred)

                ims.append(disp_im)

            distances.append(cur_distances)
            print(".", end="", flush=True)

        if visualize:
            disp_im = _combine_word_images(ims, 1024, 8, 8)
            cv2.namedWindow("positions", cv2.WINDOW_NORMAL)
            cv2.imshow("positions", disp_im)
            cv2.waitKey()

        # mean overlap - higher is better
        res = np.mean([1.0 - y for x in distances for y in x])
        return res

    return distance_test


def _visualize_position_predictions(
        word_im,
        positions_true,
        positions_pred,
        closest_idxs):

    """render an image for a single word"""

    disp_im = 255 - np.copy(word_im)

    for idx_true, idx_pred in enumerate(closest_idxs):
        pos_true = positions_true[idx_true]
        pos_pred = positions_pred[idx_pred]
        # disp_im = 255 - np.copy(word_im.data)

        disp_im[:, pos_true[0]:pos_true[1], 1] = 255
        disp_im[:, pos_true[0]] = (0, 255, 0)
        disp_im[:, pos_true[1]] = (0, 255, 0)

        disp_im[:, pos_pred[0]:pos_pred[1], 2] = 255
        disp_im[:, pos_pred[0]] = (0, 0, 255)
        disp_im[:, pos_pred[1]] = (0, 0, 255)

    return disp_im


def _visualize_position_predictions_stacked(
        word_im,
        positions_true,
        positions_pred):

    """render an image for a single word"""

    im_height = word_im.shape[0]
    double_height = im_height * 2

    disp_im = np.zeros((double_height, word_im.shape[1], 3), np.uint8)

    disp_im[0:im_height] = 255 - word_im
    disp_im[im_height:double_height] = 255 - word_im

    for pos_true in positions_true:
        # disp_im[0:im_height, pos_true[0]:pos_true[1], 1] = 255
        disp_im[0:im_height, pos_true[0]:(pos_true[0] + 1)] = (0, 255, 0)
        disp_im[0:im_height, (pos_true[1] - 1):pos_true[1]] = (0, 255, 0)

    for pos_pred in positions_pred:
        # disp_im[im_height:double_height, pos_pred[0]:pos_pred[1], 2] = 255
        disp_im[im_height:double_height, pos_pred[0]:(pos_pred[0] + 1)] = (0, 0, 255)
        disp_im[im_height:double_height, (pos_pred[1] - 1):pos_pred[1]] = (0, 0, 255)

    return disp_im


def _combine_word_images(ims, width, x_pad, y_pad):

    """combine images of the same height but different widths into a single
    image"""

    im_height = ims[0].shape[0]
    row_height = im_height + y_pad

    # build rows of images
    rows = []
    idx = 0
    row = []
    row_width = 0

    while idx < len(ims):
        im = ims[idx]
        im_width = im.shape[1]

        row_width_new = row_width + im_width + x_pad
        if row_width_new <= width:
            row.append(im)
            row_width = row_width_new
        else:
            rows.append([x for x in row])
            row = [im]
            row_width = im_width + x_pad

        idx += 1

    res_height = len(rows) * row_height
    res = np.zeros((res_height, width, 3), dtype=np.uint8)

    for row_idx, row in enumerate(rows):
        y_pos = row_idx * row_height
        x_pos = 0
        for im in row:
            res[y_pos:(y_pos + im_height), x_pos:(x_pos + im.shape[1])] = im
            x_pos += (im.shape[1] + x_pad)

    return res


def _load_words(filenames):
    """load verified word image samples from multiple files"""

    images = []
    positions = []

    for filename in filenames:
        line_poss = util.load(filename).result

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
        print()

        # save memory by ditching extracted character images etc
        for word_im in word_ims:
            images.append(word_im.data)
            positions.append([char_pos.data for char_pos in word_im.result])

    return images, positions


def _load_samples_old(filenames, half_width, offset):
    """old method of loading word image slices"""

    images = []
    combined_labels = []

    for filename in filenames:
        line_poss = util.load(filename).result

        # get word images and verified character positions
        word_ims = [word_pos.result
                    for line_pos in line_poss
                    for word_pos in line_pos.result.result
                    if word_pos.result.verified]

        # helper functions for extracting images
        # extract_char = lambda cpos, im: im[:, np.maximum(cpos[0], 0):cpos[1]]

        def extract_char_half_width(x, im):
            return improc.extract_pos(
                (x + offset - half_width, x + offset + half_width), im)

        def half(start, end):
            """point halfway between two points"""
            return int((start + end) * 0.5)

        def extract(extract_func):
            """extract images from valid char positions using extract_func"""
            res = [(extract_func(char_pos.data, word_im.data),
                    char_pos.result.result, char_pos.data, word_im.data)
                   for word_im in word_ims
                   for char_pos in word_im.result
                   if (char_pos.result.result not in IGNORE_CHARS)
                   and char_pos.result.verified
                   and (char_pos.data[1] - char_pos.data[0]) > 1]
            return res

        # extract images from the ends of all positions in each word
        char_end_ims = (
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

        # filter out images that are too small
        char_end_ims = [x for x in char_end_ims if x[0].shape[1] > 1.5 * half_width]
        char_middle_ims = [x for x in char_middle_ims if x[0].shape[1] > 1.5 * half_width]

        data_with_src = char_end_ims + char_middle_ims
        labels = [True] * len(char_end_ims) + [False] * len(char_middle_ims)

        combined_labels.append([(x, y[1]) for x, y in zip(labels, data_with_src)])
        images.append([x[0] for x in data_with_src])

    return images, combined_labels


def _load_samples(filenames, half_width, offset):
    """load word image slices from multiple files"""

    images = []
    combined_labels = []

    for filename in filenames:
        line_poss = util.load(filename).result

        # get word images and verified character positions
        word_ims = [word_pos.result
                    for line_pos in line_poss
                    for word_pos in line_pos.result.result
                    if word_pos.result.verified]

        # helper functions for extracting images
        # extract_char = lambda cpos, im: im[:, np.maximum(cpos[0], 0):cpos[1]]

        def extract_char_half_width(x, im): return improc.extract_pos(
            (x + offset - half_width, x + offset + half_width), im)

        area_true = 2

        for word_im in word_ims:

            for char_pos in word_im.result:
                if ((char_pos.result.result not in IGNORE_CHARS)
                        and char_pos.result.verified
                        and (char_pos.data[1] - char_pos.data[0]) > 1):

                    char_im = char_pos.result.data

                    for x_pos in range(0, char_im.shape[1]):

                        extract_im = extract_char_half_width(
                            char_pos.data[0] + x_pos, word_im.data)

                        # choose gap samples from start and end of each
                        # character position
                        label = (x_pos < area_true or x_pos > char_im.shape[1] - area_true - 1)

                        # choose gap samples only from start
                        # label = True if x_pos < area_true else False

                        # choose gap samples only from end
                        # label = x_pos > (char_im.shape[1] - area_true)

                        images.append(extract_im)
                        combined_labels.append((label, char_pos.result.result))

                        # cv2.namedWindow("word", cv2.WINDOW_NORMAL)
                        # cv2.namedWindow("extract", cv2.WINDOW_NORMAL)
                        # disp_word_im = np.copy(word_im.data)
                        # disp_word_im[:, char_pos.data[0] + x] = (0, 0, 255)
                        # print(char_pos.data[0] + x, label, char_pos.result.result)
                        # cv2.imshow("word", disp_word_im)
                        # cv2.imshow("extract", extract_im)
                        # cv2.waitKey(200)

    return images, combined_labels


def build_prepare_validation_callback(
        data_validate,
        labels_validate,
        build_find_prob,
        distance_test):

    def prepare_callback(feat_extractor):

        """given feature extractor functions and validation data, build callback
        to validate the network during training"""

        feats_validate = [feat_extractor(x) for x in data_validate]
        print("validation features size:", util.mbs(feats_validate), "MiB")

        def callback(classifier):
            """helper"""

            def img_to_prob(img):
                """helper"""
                res = classifier.predict_proba(feat_extractor(img))
                # probabilities are False, True in 1x2 tensor
                # so [0, 1] is the True probability
                return res[0, 1]

            print("distance test...", end="", flush=True)
            find_prob = build_find_prob(img_to_prob)
            distance = distance_test(find_prob, False)
            print("done")

            print("validation distance:", distance)

            print("predicting...", end="", flush=True)
            probs_true_pred = [classifier.predict_proba(x)[0, 1] for x in feats_validate]
            labels_validate_pred = [x > 0.5 for x in probs_true_pred]
            print("done")

            # TODO: something generic here instead of sklearn

            fpr, tpr, _ = sklearn.metrics.roc_curve(
                labels_validate, probs_true_pred)
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            print("validation ROC AUC:", roc_auc)

            accuracy = sklearn.metrics.accuracy_score(
                labels_validate, labels_validate_pred)
            print("validation accuracy:", accuracy)

            return [
                ("val_distance", distance),
                ("val_roc_auc", roc_auc),
                ("val_accuracy", accuracy)]

        return callback

    return prepare_callback


def main(argv):
    """main program"""

    if len(argv) < 2:
        mode = MODE_TUNE
    else:
        mode = argv[1]

    if len(argv) < 3:
        config = CONFIG_DEFAULT
    else:
        config = cf.load(Config, argv[2])

    if len(argv) < 4:
        model_filename = "models/classify_charpos.pkl"
    else:
        model_filename = argv[3]

    print("run_charposml")
    print("---------------")
    cf.pretty_print(config)
    print("mode:", mode)
    print("model filename:", model_filename)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    do_destructive_prepare_balance = True
    thresh_true = 0.5

    train_filenames = data.pages(config.train_idxs)
    test_filenames = data.pages(config.test_idxs)

    # for integration testing
    # train_filenames = data.pages([5, 6, 7])
    # test_filenames = data.pages([8])

    print("training files:", train_filenames)
    print("test files:", test_filenames)

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = _load_samples(
        train_filenames, config.half_width, config.offset)

    print(
        "unbalanced training data size:",
        util.mbs(data_train_unbalanced), "MiB")

    if VERBOSE:
        print(
            "training group sizes before balancing:",
            [(x[0], len(x[1]))
             for x in ml.group_by_label(
             data_train_unbalanced, labels_train_unbalanced)])

    if do_destructive_prepare_balance:
        print("destructively prepping training data for balancing")
        data_train_unbalanced, labels_train_unbalanced = ml.prepare_balance(
            data_train_unbalanced, labels_train_unbalanced, config.balance_factor)
        gc.collect()

    print(
        "prepared training data size:",
        util.mbs(data_train_unbalanced), "MiB")
    print()

    if config.do_balance:
        pad_image = partial(
            improc.pad_image,
            width=config.half_width * 2,
            height=config.pad_height)
        # balance classes in training set
        data_train, labels_train = ml.balance(
            data_train_unbalanced, labels_train_unbalanced,
            config.balance_factor,
            # lambda x: x
            pipe(
                pad_image,  # pad before rotations
                partial(
                    improc.transform_random,
                    trans_size=[config.trans_x_size, config.trans_y_size],
                    rot_size=config.rot_size,
                    scale_size=config.scale_size)))
    else:
        data_train = data_train_unbalanced
        labels_train = labels_train_unbalanced

    print("training data size:", util.mbs(data_train), "MiB")

    # load test set
    data_test, labels_test = _load_samples(
        test_filenames, config.half_width, config.offset)
    data_test, labels_test = ml.sort_by_label(data_test, labels_test)

    print("test data size:    ", util.mbs(data_test), "MiB")

    print("training count:    ", len(data_train))
    print("test count:        ", len(data_test))
    print()

    print("done")

    if VERBOSE:
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

    extract_char = improc.extract_pos

    if mode == MODE_TRAIN:

        print("training model...")

        word_ims_test, char_poss_test = _load_words(test_filenames)
        distance_test = build_distance_test(word_ims_test, char_poss_test)

        if True:
            # train a CNN

            # TODO: load separate validation data to use instead of test data
            data_validate = data_test
            labels_validate = labels_test

            def build_find_prob(img_to_prob):
                return lambda word_im: findletters.find_prob(
                    word_im, config.half_width, extract_char, img_to_prob, thresh_true)

            prepare_callback = build_prepare_validation_callback(
                data_validate,
                labels_validate,
                build_find_prob,
                distance_test
            )

            proc = imml.build_classification_process_cnn(
                data_train,
                labels_train,
                config.half_width * 2 - 8,
                config.pad_height,
                config.start_row,
                do_align=False,
                batch_size=config.batch_size,
                max_epochs=config.max_epochs,
                epoch_log_filename=model_filename + ".log.txt",
                prepare_callback=prepare_callback,
                save_model_filename=model_filename + ".wip",
                tsv_filename=model_filename + ".status")

        else:
            # traditional ML

            proc = imml.build_classification_process_charpos(
                data_train,
                labels_train,
                config.half_width * 2 - 8,
                config.pad_height,
                config.start_row)

        classify_char_pos, prep_image, feat_extractor, classifier = proc

        print("done")

        # summarize results

        feats_test = [feat_extractor(x) for x in data_test]
        labels_test_pred = [classifier(x) for x in feats_test]

        print(
            "accuracy score on test dataset:",
            sklearn.metrics.accuracy_score(
                labels_test, labels_test_pred))

        print("confusion matrix:")
        print(
            sklearn.metrics.confusion_matrix(
                labels_test, labels_test_pred, [True, False]))

        # save model
        util.save_dill(proc, model_filename)

    if mode == MODE_TUNE:

        # load model
        proc = util.load_dill(model_filename)
        classify_char_pos, prep_image, feat_extractor, classifier = proc

        # predict on test data
        feats_test = [feat_extractor(x) for x in data_test]
        labels_test_pred = [classifier(x) for x in feats_test]

        # calculate and visualize ROC AUC
        if False:
            # distances_test = model.decision_function(feats_test)
            # distances_test = classifier.model.predict_proba(feats_test)[:, 1]
            distances_test = [classifier.predict_proba(x)[0, 1] for x in feats_test]
            fpr, tpr, _ = sklearn.metrics.roc_curve(labels_test, distances_test)
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            print("ROC AUC on test dataset:", roc_auc)

            if VISUALIZE:
                # visualize ROC curve
                from matplotlib import pyplot as plt
                plt.figure()
                plt.plot(
                    fpr, tpr, color="red",
                    lw=2, label="ROC curve (area = " + str(roc_auc) + ")")
                plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel("FPR")
                plt.ylabel("TPR")
                plt.title("ROC")
                plt.legend(loc="lower right")
                plt.show()

        if False and VISUALIZE:
            # visualize result images

            # labels_test_pred = classify_char_pos(data_test)
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

        # test different position finding methods using a distance function
        # on each word

        print("loading test words...", end="", flush=True)
        word_ims_test, char_poss_test = _load_words(test_filenames)
        print("done")

        distance_test = build_distance_test(
            word_ims_test[100:150],
            char_poss_test[100:150])

        if False:
            # test the old peak-finding and connected component methods
            def build_find_thresh_peaks(peak_sigma, mean_divisor):
                """helper"""
                return partial(
                    findletters.find_thresh_peaks,
                    peak_sigma=peak_sigma,
                    mean_divisor=mean_divisor)

            res = func.grid_search(
                func.pipe(
                    build_find_thresh_peaks,
                    distance_test),
                peak_sigma=[1.0, 1.5, 2.0, 2.5],
                mean_divisor=[0.7, 1.0, 1.3, 1.4, 1.6])
            for config, score in res:
                print(
                    "peaks (",
                    config["peak_sigma"], config["mean_divisor"],
                    ") :", score)

            find_comp = lambda x: findwords.find_conc_comp(x[16:-16, :], merge=True)
            score = distance_test(find_comp, False)
            print("connected components:", score)

            find_comp_peaks = lambda word_im: findletters.find_combine(
                word_im, extract_char,
                find_comp,
                findletters.find_thresh_peaks)
            score = distance_test(find_comp_peaks)
            print("connected components + peaks:", score)

        def img_to_prob(img):
            """helper"""
            res = classifier.predict_proba(feat_extractor(img))
            # probabilities are False, True in 1x2 tensor
            # so [0, 1] is the True probability
            return res[0, 1]

        # find_comp = lambda x: findwords.find_conc_comp(x[16:-16, :], merge=True)
        # find_prob = lambda word_im: findletters.find_prob(
        #     word_im, half_width, extract_char, img_to_prob, thresh_true)
        # find_combine = lambda word_im: findletters.find_combine(
        #     word_im, extract_char,
        #     find_comp,
        #     find_prob)
        # score = distance_test(find_combine, False)
        # print("connected components + ML (", thresh_true, ") :", score)

        for thresh in [0.5, 0.6, 0.7]:  # [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
            find_prob = lambda word_im: findletters.find_prob(
                word_im, config.half_width, extract_char, img_to_prob, thresh)
            score = distance_test(find_prob, False)
            print("ML (", thresh, ") :", score)


if __name__ == "__main__":
    main(sys.argv)
