"""

Machine learning utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import random

import cv2
import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn import neural_network

from handwriting.func import grid_search, pipe


def build_feat_selection_pca(feats, n_components):
    """select features by PCA dimensionality reduction"""

    pca = PCA(n_components)
    pca.fit(feats)
    print("PCA components:", pca.n_components_, "/", pca.n_features_)
    print("PCA variance explained:", np.sum(pca.explained_variance_ratio_))

    def select(feats_test):
        """do the feature selection"""
        return pca.transform(feats_test)

    return select


def build_scaler(feats, robust=True):
    """build a feature scaler"""

    if robust:
        scaler = sklearn.preprocessing.RobustScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()

    scaler.fit(feats)

    def scale(feats_test):
        """peform scaling"""
        return scaler.transform(feats_test)

    return scale


def build_cross_validation(n_splits):

    """Build a function to perform k-fold cross validation to fit
    a model."""

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    def cross_validate(fit_func, score_func, feats, labels):

        """Perform the cross validation."""

        scores = []
        for idxs_train, idxs_test in kfold.split(feats):

            feats_train = [feats[idx] for idx in idxs_train]
            labels_train = [labels[idx] for idx in idxs_train]
            feats_test = [feats[idx] for idx in idxs_test]
            labels_test = [labels[idx] for idx in idxs_test]

            model = fit_func(
                feats_train, labels_train)

            score = score_func(model, feats_test, labels_test)

            if score is None:
                break
            scores.append(score)

        if len(scores) == n_splits:
            mean_score = np.mean(scores)
            return mean_score
        else:
            return None

    return cross_validate


def build_svc_fit(
        c, gamma, support_ratio_max):

    """Build a function that fits an RBF SVC, returning None instead of the
    model if the support ratio exceeds support_ratio_max."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""

        # verify feature min, max, and standard deviation
        # feats_mat = np.vstack(feats_train)
        # print("min:", np.min(feats_mat, axis=0))
        # print("max:", np.max(feats_mat, axis=0))
        # print("mean:", np.mean(feats_mat, axis=0))
        # print("std: ", np.std(feats_mat, axis=0))

        print(np.round(c, 4), np.round(gamma, 5), ": ", end="", flush=True)

        svc = SVC(C=c, gamma=gamma, probability=True)
        svc.fit(feats_train, labels_train)

        support_ratio = np.sum(svc.n_support_) / len(feats_train)
        if support_ratio > support_ratio_max:
            print("bad support:", np.round(support_ratio, 4), "- skipping")
            return None

        print(np.round(support_ratio, 4), end=" ")

        return svc

    return fit


def build_linear_svc_fit(c):

    """Build a function that fits a linear SVC."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""

        print(np.round(c, 4), ": ", end="", flush=True)

        svc = sklearn.svm.LinearSVC(C=c)
        svc.fit(feats_train, labels_train)

        return svc

    return fit


def build_nn_classifier(hidden_layer_sizes, alpha):

    """Build a function that fits a neural network classifier."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""

        # default model uses relu activation function, adam optimizer.
        # only loss function available is cross-entropy

        # probably will want to experiment with learning rate eventually

        print(
            hidden_layer_sizes, np.round(alpha, 4), ": ", end="", flush=True)

        model = neural_network.MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha
            # tol=-1.0e16,
            # warm_start=True,
            # max_iter=1
        )

        # for i in range(10):
        #     for j in range(10):
        #         model.fit(feats_train, labels_train)
        #     print("  ", model.n_iter_, np.round(model.loss_, 5), flush=True)
        # print("->", end=" ", flush=True)

        model.fit(feats_train, labels_train)

        return model

    return fit


def score_accuracy(model, feats_test, labels_test):

    """geneneric sklearn model scoring using accuracy"""

    if model is not None:
        labels_test_pred = model.predict(feats_test)
        score = sklearn.metrics.accuracy_score(labels_test, labels_test_pred)
        print(np.round(score, 4))
        return score
    else:
        return None


def score_auc(model, feats_test, labels_test, decision_function=True):

    """sklearn model scoring using AUC"""

    if model is not None:
        if decision_function:
            distances_test = model.decision_function(feats_test)
        else:
            # only correct for binary classifier
            predictions = model.predict_proba(feats_test)
            distances_test = predictions[:, 1]

        fpr, tpr, _ = sklearn.metrics.roc_curve(
            labels_test, distances_test)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        print(np.round(roc_auc, 4))
        return roc_auc
    else:
        return None


def train_classifier(
        fit_model, score_func, n_splits, feats, labels, **grid_search_params):

    """train a classifier"""

    cross_validate = partial(
        build_cross_validation(n_splits),
        score_func=score_func,
        feats=feats,
        labels=labels)

    fit_model_with_cross_validation = pipe(
        fit_model,
        cross_validate)

    models = grid_search(
        fit_model_with_cross_validation,
        **grid_search_params)
    models = [x for x in models if x[1] is not None]

    best_params, mean_score = models[np.argmax([x[1] for x in models])]

    print("best mean score:", mean_score)
    print("best hyperparameters:", best_params)
    model = fit_model(**best_params)(feats, labels)

    def predict(feats_test):
        """helper"""
        return model.predict(feats_test)

    return predict, model


def build_classification_process(feat_extractor, feat_selector, classifier):
    """combine feature extraction, feature selection, and classification"""

    def predict(data):
        """helper"""
        feats = feat_selector([feat_extractor(x) for x in data])
        return classifier(feats)

    return predict


def pad_image(char_bmp, width, height):
    """pad char image in a larger image"""

    xoff = abs(int((char_bmp.shape[1] - width) / 2))
    yoff = abs(int((char_bmp.shape[0] - height) / 2))

    if width >= char_bmp.shape[1]:
        x_min_old = 0
        x_max_old = char_bmp.shape[1]
        x_min_new = xoff
        x_max_new = char_bmp.shape[1] + xoff
    else:
        x_min_old = xoff
        x_max_old = width + xoff
        x_min_new = 0
        x_max_new = width

    if height >= char_bmp.shape[0]:
        y_min_old = 0
        y_max_old = char_bmp.shape[0]
        y_min_new = yoff
        y_max_new = char_bmp.shape[0] + yoff
    else:
        y_min_old = yoff
        y_max_old = height + yoff
        y_min_new = 0
        y_max_new = height

    image_subset = char_bmp[y_min_old:y_max_old, x_min_old:x_max_old]
    new_bmp = np.ones((height, width, 3), dtype=np.uint8) * 255
    new_bmp[y_min_new:y_max_new, x_min_new:x_max_new] = image_subset

    return new_bmp


def filter_cc(image):
    """find connected components in a threshold image and white out
    everything except the second largest"""

    # TODO: better way to select relevant components
    comp_filt = np.copy(image)
    gray = 255 - np.array(np.sum(image, axis=2) / 3.0, dtype=np.uint8)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    connectivity = 4
    comps = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    labels = comps[1]
    sizes = comps[2][:, cv2.CC_STAT_AREA]

    # get index of second-largest component
    if len(sizes) > 1:
        second_largest_idx = np.argsort(sizes)[-2]
    else:
        second_largest_idx = np.argsort(sizes)[-1]

    # eliminate everything else
    for label_idx in range(len(sizes)):
        if label_idx != second_largest_idx:
            comp_filt[labels == label_idx] = 255

    # cv2.imshow("image", image)
    # cv2.imshow("gray", gray)
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("comp_filt", comp_filt)
    # cv2.waitKey()

    return comp_filt


def align(image):
    """shift an image so the center of mass of the pixels is centered"""

    # TODO: this should just operate on grayscale

    gray = 255 - np.array(np.sum(image, axis=2) / 3.0, dtype=np.uint8)

    x_size = image.shape[1]
    y_size = image.shape[0]
    x_mean = np.sum(np.sum(gray, axis=0) * np.arange(x_size)) / np.sum(gray)
    y_mean = np.sum(np.sum(gray, axis=1) * np.arange(y_size)) / np.sum(gray)

    tmat = np.float32([[1, 0, x_size / 2.0 - x_mean], [0, 1, y_size / 2.0 - y_mean]])
    new_image = cv2.warpAffine(
        image, tmat, (image.shape[1], image.shape[0]), borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", new_image)
    # cv2.waitKey()

    return new_image


def grayscale(image):
    """convert RGB ubyte image to grayscale"""
    return np.sum(image, axis=2) / 3.0


def downsample(image, scale_factor):
    """downsample an image and unravel to create a feature vector"""

    feats = cv2.resize(
        image,
        (int(image.shape[0] * scale_factor),
         int(image.shape[1] * scale_factor)))
    return feats


def downsample_4(image):
    """create a feature vector from four downsampling amounts"""

    return downsample_multi(image, [0.4, 0.2, 0.1, 0.05])


def downsample_multi(image, scales):
    """create a feature vector from arbitrary downsampling amounts"""

    return np.hstack([np.ravel(downsample(image, x)) for x in scales])


def max_pool(im):
    """perform 2x2 max pooling"""

    return np.max(
        np.stack(
            (im[0::2, 0::2],
             im[0::2, 1::2],
             im[1::2, 0::2],
             im[1::2, 1::2]),
            axis=-1),
        axis=-1)


def max_pool_multi(image, ns):
    """perform multiple levels of max pooling and unravel
    to create a feature vector"""

    # TODO: move this to a higher level
    # image_gray = _grayscale(image)

    if 1 in ns:
        res = [image]
    else:
        res = []
    for n in range(2, max(ns) + 1):
        image = max_pool(image)
        if n in ns:
            res.append(image)
    return np.hstack([np.ravel(y) for y in res])


def column_ex(gray):

    """experimental feature - something like the center of mass of
    overlapping columns of the image"""

    width = 2
    # mul_mat = np.arange(y_size)[:, np.newaxis]
    # for some reason, it works a lot better to not divide by the sum of the
    # whole window but only the first column.
    mul_mat = np.linspace(0, 1, gray.shape[0])[:, np.newaxis]

    y_agg = np.array([(np.sum(gray[:, idx + width] * mul_mat) /
                       np.sum(gray[:, idx]))
                      for idx in range(gray.shape[1] - width)])
    y_agg[~np.isfinite(y_agg)] = 0.0

    res = np.hstack((y_agg, np.diff(y_agg)))

    return res


#def column_ex_im(gray):
#
#    """image version of column_ex"""
#
#    y_size = gray.shape[0] * 1.0
#    y_agg = np.array([np.sum(gray[:, idx:(idx + 3)] * np.arange(y_size)[:, np.newaxis] / np.sum(gray[:, idx]))
#                      for idx in range(gray.shape[1] - 3)])
#    y_agg[np.isnan(y_agg)] = y_size
#    # data = np.array(np.clip(y_agg, 0, y_size - 2), dtype=np.int)
#    data = np.array(
#        np.clip(np.diff(y_agg) + y_size / 2.0, 0, y_size - 2), dtype=np.int)
#
#    res_im = np.zeros(gray.shape)
#    for idx in range(data.shape[0]):
#        y_val = data[idx]
#        res_im[y_val, idx] = 1.0
#        res_im[y_val + 1, idx] = 1.0
#
#    return res_im


def group_by_label(samples, labels):
    """group samples by label"""

    unique_labels = sorted(list(set(labels)))

    grouped = [[x for x, y in zip(samples, labels) if y == label]
               for label in unique_labels]

    res_unsorted = list(zip(unique_labels, grouped))
    # order = np.argsort([len(x) for x in grouped])
    order = sorted(range(len(grouped)), key=lambda k: unique_labels[k])
    return [res_unsorted[idx] for idx in order]


def balance(samples, labels, balance_factor, adjust_func):
    """create a balanced dataset by subsampling classes or generating new
    samples"""

    grouped = group_by_label(samples, labels)

    if balance_factor <= 1.0:
        largest_group_size = max([len(x[1]) for x in grouped])
        target_group_size = int(largest_group_size * balance_factor)
    else:
        target_group_size = int(balance_factor)

    grouped_balanced = []
    for label, group in grouped:

        if len(group) > target_group_size:
            print(label, 1.0)
            group_resized = random.sample(group, target_group_size)
        else:
            print(label, (len(group) * 1.0) / target_group_size)
            group_resized = [x for x in group]
            while len(group_resized) < target_group_size:
                group_resized.append(adjust_func(random.choice(group)))
        grouped_balanced.append((label, group_resized))

    pairs = [(y, x[0]) for x in grouped_balanced for y in x[1]]
    return zip(*pairs)


def transform_random(image, trans_size, rot_size, scale_size):
    """apply a small random transformation to an image"""

    # TODO: make ranges of random numbers input parameters
    trans = np.random.rand(2) * trans_size - 0.5 * trans_size
    rot = np.random.rand(4) * rot_size - 0.5 * rot_size
    scale = 1.0 + np.random.rand(1)[0] * scale_size - 0.5 * scale_size

    x_size = image.shape[1]
    y_size = image.shape[0]

    trans_to_center = np.float32(
        [[1, 0, -x_size / 2.0],
         [0, 1, -y_size / 2.0],
         [0, 0, 1]])
    trans_from_center = np.float32(
        [[1, 0, x_size / 2.0],
         [0, 1, y_size / 2.0],
         [0, 0, 1]])
    trans_random = np.float32(
        [[1 + rot[0], 0 + rot[1], trans[0]],
         [0 + rot[2], 1 + rot[3], trans[1]],
         [0, 0, 1]])
    trans_scale = np.identity(3, dtype=np.float32) * scale

    tmat = np.dot(trans_from_center, np.dot(trans_scale, np.dot(trans_random, trans_to_center)))[0:2, :]

    image_new = cv2.warpAffine(
        image, tmat,
        (image.shape[1], image.shape[0]),
        borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", image_new)
    # cv2.waitKey()

    return image_new
