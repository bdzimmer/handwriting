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

import cv2
import numpy as np
import sklearn

from handwriting import util, charclass, ml, func, improc, cnn
from handwriting import findletters, findwords
from handwriting import data
from handwriting.prediction import Sample

VISUALIZE = False
VERBOSE = False

MODE_TRAIN = "train"
MODE_TUNE = "tune"


def build_classification_process(
        data_train,
        labels_train,
        pad_width,
        prepare_callback=None):

    """build a classification process"""

    start_row = 0 # 32

    pad_image = partial(improc.pad_image, width=pad_width, height=96)

    def prep_image(image):
        """prepare an image (result is still a 2d image)"""
        image = image[start_row:, :]
        # return 255.0 - improc.grayscale(
        #     improc.align(pad_image(image), x_align=False))
        return 255.0 - improc.grayscale(pad_image(image))

    if VISUALIZE:
        # visualize training data
        for cur_label, group in ml.group_by_label(data_train, labels_train):
            print("label:", cur_label)
            group_prepped = [(prep_image(x), None) for x in group]
            # print(np.min(group_prepped[0][0]), np.max(group_prepped[0][0]))
            group_pred = [Sample(x, cur_label, 0.0, False) for x in group_prepped]
            chars_working, chars_done = charclass.label_chars(group_pred)

    if False:

        # training process for traditional machine learning models
        # with cross validation

        def feat_extractor(image):
            """convert image to feature vector"""
            img_p = prep_image(image)
            img_g = img_p / 255.0
            img_g = img_g / np.max(img_g)

            grad_0, grad_1 = np.gradient(img_g)
            return np.hstack((
                improc.max_pool_multi(grad_0, [2]),
                improc.max_pool_multi(grad_1, [2]),
                improc.max_pool_multi(img_g, [2])))

        feats_train = [feat_extractor(x) for x in data_train]
        # feat_selector = ml.build_feat_selection_pca(feats_train, 0.99)
        # feat_selector = ml.build_feat_selection_pca(feats_train, 0.99)
        feat_selector = lambda x: x
        # feat_selector = ml.build_scaler(feats_train, robust=True)

        feats_train = feat_selector(feats_train)
        print("feature length:", len(feats_train[0]))

        # classifier = ml.train_classifier(
        #     build_fit_model=ml.linear_svc,
        #     cross_validation=ml.kfold_cross_validation(5),
        #     score_func=partial(ml.score_auc, decision_function=True),
        #     feats=feats_train,
        #     labels=labels_train,
        #     gamma=np.logspace(-2, 0, 20),
        # )

        classifier = ml.train_classifier(
            build_fit_model=ml.nn_classifier,
            cross_validation=ml.kfold_cross_validation(10),
            score_func=partial(ml.score_auc, decision_function=False),
            feats=feats_train,
            labels=labels_train,
            # hidden_layer_sizes=[(16, 16), (32, 32), (256, 128), (256, 64), (256, 32)],
            # hidden_layer_sizes=[(128, 128, 128), (256, 256, 256)],
            # hidden_layer_sizes=[(128,), (256,), (128, 128), (256, 256)],
            hidden_layer_sizes=[(128, 128, 128), (128, 128, 128, 128), (64, 64), (64, 64, 64), (64, 64, 64, 64)],
            # alpha=[0.0001, 0.01]
            alpha=[0.0001, 0.001, 0.01]
        )
    else:

        # training process for deep neural networks

        do_lazy_extraction = True

        callbacks_log_filename = (
            "log_charpos_callback.txt"
            if prepare_callback is not None
            else None)

        if not do_lazy_extraction:

            def feat_extractor(image):
                """convert image to feature vector"""
                img_p = prep_image(image)
                img_g = img_p / 255.0 - 0.5
                return np.array(img_g, np.float32) # Torch wants float32

            feats_train = [feat_extractor(x) for x in data_train]
            feat_selector = lambda x: x
            feats_train = feat_selector(feats_train)

            print("preparing callback...", end="", flush=True)
            callback = prepare_callback(feat_extractor, feat_selector)
            print("done")

            lazy_extractor = None

        else:
            # trade memory for compute

            def color_to_grayuint(image):
                """prepare image to ubyte"""
                image = image[start_row:, :]
                return np.array(
                    255.0 - improc.grayscale(pad_image(image)),
                    dtype=np.uint8)

            def grayuint_to_grayfloat(image):
                """convert uint8 image to floating point"""
                img_g = image / 255.0 - 0.5
                return np.array(img_g, np.float32)

            feat_extractor = lambda x: grayuint_to_grayfloat(color_to_grayuint(x))
            feat_selector = lambda x: x

            # this is very small
            feats_train = [color_to_grayuint(x) for x in data_train]
            del data_train
            gc.collect()

            print("preparing callback...", end="", flush=True)
            callback = prepare_callback(
                feat_extractor, feat_selector)
            print("done")

            lazy_extractor = grayuint_to_grayfloat

        print("training features (input to CNN) size:", mbs(feats_train), "MiB")

        callbacks_log_filename = (
            "log_charpos_callback.txt"
            if prepare_callback is not None
            else None)

        # prepare the callback...this is a little awkward
        # I think the solution is probably to return all of the pieces
        # along with a function which trains the classifier given callbacks
        # or something like that

        classifier = cnn.experimental_cnn(
            batch_size=16, # 8
            max_epochs=12,
            learning_rate=0.001,
            momentum=0.9,
            epoch_log_filename="log_charpos.txt",
            callback_log_filename=callbacks_log_filename,
            callback=callback,
            callback_rate=4,
            lazy_extractor=lazy_extractor
        )(
            feats_train,
            labels_train
        )

    classify_char_image = ml.classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier)


def _visualize_position_predictions(
        word_im,
        positions_true,
        positions_pred,
        distances,
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


def build_distance_test(word_ims_test, char_poss_test):
    """create a distance test function to measure jaccard distance
    for various methods"""

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
                #     distances,
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

        # TODO: should I be aggregating this differently?
        # mean overlap - higher is better
        res = np.mean([1.0 - y for x in distances for y in x])
        return res

    return distance_test


def _load_words(filenames):
    """load verified word image samples from multiple files"""

    # TODO: move into loop to save memory

    images = []
    positions = []

    for filename in filenames:
        line_poss = util.load(filename).result

        # load multiple files
        # line_poss = [y for x in filenames for y in util.load(x).result]

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

        # save memory by ditching extracted character images etc
        for word_im in word_ims:
            images.append(word_im.data)
            positions.append([char_pos.data for char_pos in word_im.result])

    return images, positions


def _load_samples(filenames, half_width, offset):
    """load word image slices from multiple files"""

    # ignore_chars = ["`", "~", "\\", "/"]
    ignore_chars = []

    images = []
    combined_labels = []

    for filename in filenames:
        line_poss = util.load(filename).result

        # load multiple files
        # line_poss = [y for x in filenames for y in util.load(x).result]

        # get word images and verified character positions
        word_ims = [word_pos.result
                    for line_pos in line_poss
                    for word_pos in line_pos.result.result
                    if word_pos.result.verified]

        # helper functions for extracting images
        # extract_char = lambda cpos, im: im[:, np.maximum(cpos[0], 0):cpos[1]]

        extract_char_half_width = lambda x, im: improc.extract_pos(
            (x + offset - half_width, x + offset + half_width), im)

        if False:

            def half(start, end):
                """point halfway between two points"""
                return int((start + end) * 0.5)

            def extract(extract_func):
                """extract images from valid char positions using extract_func"""
                res = [(extract_func(char_pos.data, word_im.data),
                        char_pos.result.result, char_pos.data, word_im.data)
                       for word_im in word_ims
                       for char_pos in word_im.result
                       if (char_pos.result.result not in ignore_chars)
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

        else:

            area_true = 2

            for word_im in word_ims:

                for char_pos in word_im.result:
                    if ((char_pos.result.result not in ignore_chars)
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

                            combined_labels.append(
                                (label,
                                 char_pos.result.result))

                            # cv2.namedWindow("word", cv2.WINDOW_NORMAL)
                            # cv2.namedWindow("extract", cv2.WINDOW_NORMAL)
                            # disp_word_im = np.copy(word_im.data)
                            # disp_word_im[:, char_pos.data[0] + x] = (0, 0, 255)
                            # print(char_pos.data[0] + x, label, char_pos.result.result)
                            # cv2.imshow("word", disp_word_im)
                            # cv2.imshow("extract", extract_im)
                            # cv2.waitKey(200)

    return images, combined_labels


def mbs(arrays):
    """find the approximate size of a list of numpy arrays in MiB"""
    total = 0.0
    for array in arrays:
        total += array.nbytes / 1048576.0
    return np.round(total, 3)


def main(argv):
    """main program"""

    if len(argv) < 2:
        mode = MODE_TUNE
    else:
        mode = argv[1]

    np.random.seed(0)
    random.seed(0)

    model_filename = "models/classify_charpos.pkl"
    half_width = 16 # 16
    offset = 0
    do_destructive_prepare_balance = True
    do_balance = False
    balance_factor = 1024 # 256 # 128

    train_filenames, test_filenames = data.pages([5, 6, 7, 9, 10, 11, 12], [8])

    # train_filenames, test_filenames = data.pages([0, 1, 2, 3, 4, 5, 6], [7, 8])
    # train_filenames, test_filenames = data.pages([5, 6, 9, 10, 11, 12], [7, 8])
    # train_filenames, test_filenames = data.pages([0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12], [7, 8])


    # train_filenames, test_filenames = data.pages([5, 6, 7, 9, 10], [8])
    # train_filenames, test_filenames = data.pages([5, 6, 7], [8])

    print("training files:", train_filenames)
    print("test files:", test_filenames)

    print("loading and balancing datasets...")

    # import gc
    # gc.collect()

    # load training set
    data_train_unbalanced, labels_train_unbalanced = _load_samples(
        train_filenames, half_width, offset)

    # print("training data shapes:", sorted(list(set([x.shape for x in data_train_unbalanced]))))
    # print("training data length:", len(data_train_unbalanced))

    print("unbalanced training data size:", mbs(data_train_unbalanced), "MiB")

    if VERBOSE:
        print(
            "training group sizes before balancing:",
            [(x[0], len(x[1]))
             for x in ml.group_by_label(
             data_train_unbalanced, labels_train_unbalanced)])

    if do_destructive_prepare_balance:
        print("destructively prepping training data for balancing")
        data_train_unbalanced, labels_train_unbalanced = ml.prepare_balance(
            data_train_unbalanced, labels_train_unbalanced, balance_factor)
        gc.collect()

    print("prepared training data size:", mbs(data_train_unbalanced), "MiB")
    print()

    if do_balance:

        # balance classes in training set
        data_train, labels_train = ml.balance(
            data_train_unbalanced, labels_train_unbalanced,
            balance_factor,
            lambda x: x
            # partial(
            #     improc.transform_random,
            #     trans_size=[0.0, 12.0],
            #     rot_size=0.25,    # 0.2
            #     scale_size=0.25  # 0.1
            # )
        )

    else:

        data_train = data_train_unbalanced
        labels_train = labels_train_unbalanced

    print("training data size:", mbs(data_train), "MiB")

    # load test set
    data_test, labels_test = _load_samples(test_filenames, half_width, offset)

    # the purpose of this is to group the test data and sort it for easier
    # visualization later
    test_gr = dict(ml.group_by_label(data_test, labels_test))
    test_grf = test_gr
    data_test, labels_test = zip(*[
        (y, x[0]) for x in test_grf.items() for y in x[1]])

    print("test data size:    ", mbs(data_test), "MiB")

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

    # rebalance between true and false
    # data_train, labels_train = ml.balance(
    #     data_train, labels_train,
    #     2000,
    #     lambda x: x)

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_train, labels_train)])
    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in ml.group_by_label(data_test, labels_test)])

    # extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]
    extract_char = improc.extract_pos

    if mode == MODE_TRAIN:

        print("training model...")

        word_ims_test, char_poss_test = _load_words(test_filenames)
        distance_test = build_distance_test(word_ims_test, char_poss_test)

        def prepare_callback(feat_extractor, feat_selector):
            """given feature extractor and feature selector functions,
            build callbacks to test the network during training"""
            feats_test = feat_selector([feat_extractor(x) for x in data_test])

            print("validation features size:", mbs(feats_test), "MiB")

            def callback(model):
                """helper"""
                print("predicting...", end="", flush=True)
                probs_true_pred = [x[0, 1] for x in model.predict_proba(feats_test)]
                labels_test_pred = [x > 0.5 for x in probs_true_pred]
                print("done")

                def classify_ml(img):
                    """helper"""
                    res = model.predict_proba(
                        feat_selector([feat_extractor(y) for y in img]))
                    # probabilities are False, True in 1x2 tensor
                    # so [0, 1] is the True probability
                    res = [x[0, 1] for x in res]
                    return res

                thresh = 0.5
                find_classify = lambda word_im: findletters.find_classify_prob(
                    word_im, half_width, extract_char, classify_ml, thresh)
                distance = distance_test(find_classify, False)
                print("validation distance:", distance)

                fpr, tpr, _ = sklearn.metrics.roc_curve(
                    labels_test, probs_true_pred)
                roc_auc = sklearn.metrics.auc(fpr, tpr)
                print("validation ROC AUC:", roc_auc)

                # TODO: something generic here instead of sklearn
                accuracy = sklearn.metrics.accuracy_score(
                    labels_test, labels_test_pred)
                print("validation accuracy:", accuracy)

                return [distance, roc_auc, accuracy]

            return callback

        proc = build_classification_process(
            data_train,
            labels_train,
            pad_width=half_width * 2 - 8,
            prepare_callback=prepare_callback)

        (classify_char_pos,
         prep_image, feat_extractor, feat_selector,
         classifier) = proc

        print("done")

        feats_test = feat_selector([feat_extractor(x) for x in data_test])
        # print("score on test dataset:", classifier_score(feats_test, labels_test))
        # labels_test_pred = classify_char_pos(data_test)
        labels_test_pred = classifier(feats_test)
        print("accuracy score on test dataset:", sklearn.metrics.accuracy_score(
            labels_test, labels_test_pred))

        print("confusion matrix:")
        print(sklearn.metrics.confusion_matrix(
            labels_test, labels_test_pred, [True, False]))

        if False:
            # distances_test = model.decision_function(feats_test)
            distances_test = classifier.model.predict_proba(feats_test)[:, 1]
            fpr, tpr, _ = sklearn.metrics.roc_curve(
                labels_test, distances_test)
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

        util.save_dill(proc, model_filename)

    if mode == MODE_TUNE:

        # test different position finding methods using a distance function
        # on each word

        print("loading test words...", end="", flush=True)
        word_ims_test, char_poss_test = _load_words(test_filenames)
        print("done")

        distance_test = build_distance_test(word_ims_test[100:150], char_poss_test[100:150])

        if False:
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

        # load ML model
        proc = util.load_dill(model_filename)
        (classify_char_pos,
         prep_image, feat_extractor, feat_selector,
         classifier) = proc

        # def classify_ml(img):
        #     """helper"""
        #     # select the column of true probabilities
        #     # (column 1) from the result array
        #     res = classifier.model.predict_proba(
        #         feat_selector([feat_extractor(y) for y in img]))[:, 1]
        #    return res

        def classify_ml(img):
            """helper"""
            res = classifier.predict_proba(
                feat_selector([feat_extractor(y) for y in img]))
            # probabilities are False, True in 1x2 tensor
            # so [0, 1] is the True probability
            res = [x[0, 1] for x in res]
            return res

        # thresh = 0.8
        # find_classify = lambda word_im: findletters.find_classify_prob(
        #      word_im, half_width, extract_char, classify_ml, thresh)
        # find_comp_classify = lambda word_im: findletters.find_combine(
        #     word_im, extract_char,
        #     find_comp,
        #     find_classify)
        # score = distance_test(find_comp_classify, False)
        # print("connected components + ML (", thresh, ") :", score)

        for thresh in [0.5]: # [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
            find_classify = lambda word_im: findletters.find_classify_prob(
                word_im, half_width, extract_char, classify_ml, thresh)
            score = distance_test(find_classify, True)
            print("ML (", thresh, ") :", score)


if __name__ == "__main__":
    main(sys.argv)
