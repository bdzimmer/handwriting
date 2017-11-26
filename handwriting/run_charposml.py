# -*- coding: utf-8 -*-
"""

Train a classifier to classify vertical slices of a word image as within a
a letter or between letters.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import random
import sys

import cv2
import numpy as np
import sklearn

from handwriting import util, charclass, ml, func
from handwriting import findletters, findwords
from handwriting import data
from handwriting.prediction import Sample

VISUALIZE = False


def build_classification_process(data_train, labels_train):

    """build a classification process"""

    pad_width = 16
    start_row = 32

    pad_image = partial(ml.pad_image, width=pad_width, height=96)

    def prep_image(image):
        """prepare an image (result is still a 2d image)"""
        image = image[start_row:, :]
        # return 255.0 - ml.grayscale(ml.align(pad_image(image)))
        return 255.0 - ml.grayscale(pad_image(image))

    def feat_extractor(image):
        """convert image to feature vector"""
        img_p = prep_image(image)
        img_g = img_p / 255.0
        # return np.ravel(img_g)
        # return np.ravel(ml.max_pool(img_g))

        # 2017-11-13: 3rd best
        # return ml.max_pool_multi(img_g, [1, 2])
        # return ml.downsample_4(img_g)
        # return ml.downsample_multi(img_g, [1.0, 0.5])
        # return ml.downsample_multi(img_g, [0.5, 0.25])

        # return np.ravel(ml.column_agg(img_g))
        # return np.hstack(
        #     (ml.column_ex(img_g),
        #      ml.column_ex(ml.max_pool(img_g))))

        # 2017-11-13: 2nd best
        grad_0, grad_1 = np.gradient(img_g)
        return np.hstack((
            ml.max_pool_multi(grad_0, [2]),
            ml.max_pool_multi(grad_1, [2])))

        # grad_0, grad_1 = np.gradient(ml.downsample(img_g, 0.5))
        # return np.hstack((np.ravel(grad_0), np.ravel(grad_1)))

        # 2017-11-13: best!
        # img_b = np.array(img_g * 255, dtype=np.uint8)
        # hog = cv2.HOGDescriptor(
        #     (16, 104), (8, 8), (8, 8), (2, 2), 8)
        # return np.ravel(hog.compute(img_b))

    if VISUALIZE:
        # visualize training data
        for cur_label, group in ml.group_by_label(data_train, labels_train):
            print("label:", cur_label)
            group_prepped = [(prep_image(x), None) for x in group]
            # print(np.min(group_prepped[0][0]), np.max(group_prepped[0][0]))
            group_pred = [Sample(x, cur_label, 0.0, False) for x in group_prepped]
            chars_working, chars_done = charclass.label_chars(group_pred)

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = ml.build_feat_selection_pca(feats_train, 0.99) # 0.90
    # feat_selector = lambda x: x
    # feat_selector = ml.build_scaler(feats_train, robust=True)

    feats_train = feat_selector(feats_train)
    print("feature length:", len(feats_train[0]))

    # classifier, classifier_score = ml.train_classifier(
    #     # fit_model=partial(
    #     #     ml.build_svc_fit,
    #     #     support_ratio_max=1.0),
    #     fit_model=ml.build_linear_svc_fit,
    #     # score_func=ml.score_accuracy,
    #     score_func=ml.score_auc,
    #     n_splits=5,
    #     feats=feats_train,
    #     labels=labels_train,
    #     c=np.logspace(-4, 0, 30),
    #     # gamma=np.logspace(-2, 0, 20)
    # )

    classifier, classifier_score = ml.train_classifier(
        fit_model=ml.build_nn_classifier,
        score_func=partial(ml.score_auc, decision_function=False),
        n_splits=5,
        feats=feats_train,
        labels=labels_train,
        hidden_layer_sizes=[(256, 128), (256, 64), (256, 32)],
        alpha=[0.0001]
    )

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

    # TODO: double check annotations of periods and quote marks
    ignore_chars = ["`", "~"]

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
    balance_factor = 32 # 64 # 20 # 15 # 500

    train_filenames, test_filenames = data.train_test_pages([5, 6])

    print("training files:", train_filenames)
    print("test files:", test_filenames)

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
        balance_factor,
        partial(
            ml.transform_random,
            trans_size=2.0, rot_size=0.2))

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

    if mode == "train":

        print("training model...")

        proc = build_classification_process(
            data_train, labels_train)

        (classify_char_pos,
         prep_image, feat_extractor, feat_selector,
         classifier, model) = proc

        print("done")

        feats_test = feat_selector([feat_extractor(x) for x in data_test])
        # print("score on test dataset:", classifier_score(feats_test, labels_test))
        # labels_test_pred = classify_char_pos(data_test)
        labels_test_pred = model.predict(feats_test)
        print("accuracy score on test dataset:", sklearn.metrics.accuracy_score(
            labels_test, labels_test_pred))

        print("confusion matrix:")
        print(sklearn.metrics.confusion_matrix(
            labels_test, labels_test_pred, [True, False]))

        # distances_test = model.decision_function(feats_test)
        distances_test = model.predict_proba(feats_test)[:, 1]
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            labels_test, distances_test)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print("ROC AUC on test dataset:", roc_auc)

        util.save_dill(proc, model_filename)

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

    if mode == "tune":

        # test different position finding methods using a distance function
        # on each word

        extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]

        print("loading test words...", end="", flush=True)
        word_ims_test = _load_words(test_filenames)
        print("done")

        def distance_test(find_char_poss_func, visualize=False):
            """helper"""
            jaccard_distance = lambda x, y: 1.0 - findletters.jaccard_index(x, y)
            distances = []
            for word_im in word_ims_test:
                positions_true = [x.data for x in word_im.result]
                positions_test = find_char_poss_func(word_im.data)
                cur_distances, idxs = findletters.position_list_distance(
                    positions_true, positions_test,
                    jaccard_distance)

                if visualize:
                    for idx_true, idx_test in enumerate(idxs):
                        pos_true = positions_true[idx_true]
                        pos_test = positions_test[idx_test]
                        disp_im = 255 - np.copy(word_im.data)

                        disp_im[:, pos_true[0]:pos_true[1], 0] = 255
                        disp_im[:, pos_true[0]] = (255, 0, 0)
                        disp_im[:, pos_true[1]] = (255, 0, 0)

                        disp_im[:, pos_test[0]:pos_test[1], 2] = 255
                        disp_im[:, pos_test[0]] = (0, 0, 255)
                        disp_im[:, pos_test[1]] = (0, 0, 255)

                        cv2.namedWindow("positions", cv2.WINDOW_NORMAL)
                        cv2.imshow("positions", disp_im)
                        print(
                            pos_true, pos_test,
                            cur_distances[idx_true],
                            1.0 - cur_distances[idx_true])
                        cv2.waitKey()

                distances.append(cur_distances)

            # TODO: should I be aggregating this differently?
            # mean overlap - higher is better
            res = np.mean([1.0 - y for x in distances for y in x])
            return res

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
            peak_sigma = [1.0, 1.5, 2.0, 2.5],
            mean_divisor = [0.7, 1.0, 1.3, 1.4, 1.6])
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
         classifier, model) = proc

        def classify_ml(im):
            """helper"""
            res = model.predict_proba(
                feat_selector([feat_extractor(y) for y in im]))[:, 1]
            return res

        thresh = 0.8
        find_classify = lambda word_im: findletters.find_classify_prob(
                word_im, half_width, extract_char, classify_ml, thresh)
        find_comp_classify = lambda word_im: findletters.find_combine(
            word_im, extract_char,
            find_comp,
            find_classify)
        score = distance_test(find_comp_classify, False)
        print("connected components + ML (", thresh, ") :", score)

        for thresh in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]:
            find_classify = lambda word_im: findletters.find_classify_prob(
                word_im, half_width, extract_char, classify_ml, thresh)
            score = distance_test(find_classify, False)
            print("ML (", thresh, ") :", score)


if __name__ == "__main__":
    main(sys.argv)
