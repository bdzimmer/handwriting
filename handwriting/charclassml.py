"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import itertools
import random

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC


def build_current_best_process(
        data_train, labels_train, support_ratio_max):

    """build the current best character classification process"""

    pad_image_96 = lambda x: pad_image(x, 96, 96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        start_row = 16
        image = image[start_row:, :]
        return _align(_filter_cc(pad_image_96(image)))

    def feat_extractor(image):
        """convert image to feature vector"""
        p_img = prep_image(image)
        return _downsample_4(p_img)

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = build_feat_selection_pca(feats_train, 0.90)
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = train_char_class_svc(
        feats_train, labels_train, 4, support_ratio_max)

    classify_char_image = build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)


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


def train_char_class_svc(
        feats, labels, n_splits, support_ratio_max):
    """train an SVC character classifier"""

    # train SVC, tuning hyperparameters with k-fold cross validation
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    models = []
    hyps = itertools.product(
        np.logspace(1, 2, 4),   # np.logspace(0, 2, 3)
        np.logspace(-2, 0, 5))  # np.logspace(-3, 1, 5)

    for c, gamma in hyps:
        svc_scores = []
        for idxs_train, idxs_test in kfold.split(feats):

            feats_train = [feats[idx] for idx in idxs_train]
            labels_train = [labels[idx] for idx in idxs_train]
            feats_test = [feats[idx] for idx in idxs_test]
            labels_test = [labels[idx] for idx in idxs_test]

            print(np.round(c, 3), np.round(gamma, 3), ": ", end="", flush=True)

            svc = SVC(C=c, gamma=gamma, probability=True)
            svc.fit(feats_train, labels_train)

            support_ratio = np.sum(svc.n_support_) / len(idxs_train)
            if support_ratio > support_ratio_max:
                print("bad support:", np.round(support_ratio, 4), "- skipping")
                break

            svc_score = svc.score(feats_test, labels_test)
            print(np.round(svc_score, 4), np.round(support_ratio, 4))
            svc_scores.append(svc_score)

        if len(svc_scores) == n_splits:
            mean_svc_score = np.mean(svc_scores)
            models.append(((c, gamma), mean_svc_score))

    # fit a model on all training data using the best parameters
    (c, gamma), mean_svc_score = models[np.argmax([x[1] for x in models])]
    print("best mean svc score:", mean_svc_score)
    print("best hyperparameters:", c, gamma)
    svc = SVC(C=c, gamma=gamma, probability=True)
    svc.fit(feats, labels)
    print("support vector counts:", list(zip(svc.classes_, svc.n_support_)))

    def predict(feats_test):
        """helper"""
        return svc.predict(feats_test)

    # TODO: write a general score function in the future
    def score(feats_test, labels_test):
        """helper"""
        return svc.score(feats_test, labels_test)

    return predict, score


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
        y_max_old = width + yoff
        y_min_new = 0
        y_max_new = height

    image_subset = char_bmp[y_min_old:y_max_old, x_min_old:x_max_old]
    new_bmp = np.ones((height, width, 3), dtype=np.uint8) * 255
    new_bmp[y_min_new:y_max_new, x_min_new:x_max_new] = image_subset

    return new_bmp


def _filter_cc(image):
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


def _align(image):
    """shift an image so the center of mass of the pixels is centered"""

    gray = 255 - np.array(np.sum(image, axis=2) / 3.0, dtype=np.uint8)

    x_size = image.shape[1]
    y_size = image.shape[0]
    x_mean = np.sum(np.sum(gray, axis=0) * np.arange(x_size)) / np.sum(gray)
    y_mean = np.sum(np.sum(gray, axis=1) * np.arange(y_size)) / np.sum(gray)

    tmat = np.float32([[1, 0, x_size / 2.0 - x_mean], [0, 1, y_size / 2.0 - y_mean]])
    new_image = cv2.warpAffine(image, tmat, image.shape[0:2], borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", new_image)
    # cv2.waitKey()

    return new_image


def _downsample(image, scale_factor):
    """downsample an image and unravel to create a feature vector"""

    feats = np.ravel(
        cv2.resize(
            np.sum(image, axis=2) / (255.0 * 3.0),
            (int(image.shape[0] * scale_factor),
             int(image.shape[1] * scale_factor))))
    return feats


def _downsample_4(image):
    """create a feature vector from four downsampling amounts"""
    return np.hstack((
        _downsample(image, 0.4),
        _downsample(image, 0.2),
        _downsample(image, 0.1),
        _downsample(image, 0.05)))


def _downsample_multi(image, scales):
    """create a feature vector from arbitrary downsampling amounts"""
    return np.hstack([_downsample(image, x) for x in scales])


def _max_pool(im):
    """perform 2x2 max pooling"""

    return np.max(
        np.stack(
            (im[0::2, 0::2],
             im[0::2, 1::2],
             im[1::2, 0::2],
             im[1::2, 1::2]),
            axis=-1),
        axis=-1)


def _max_pool_multi(im, ns):
    """perform multiple levels of max pooling and unravel
    to create a feature vector"""

    im = np.sum(im, axis=2) / (255.0 * 3.0)
    if 1 in ns:
        res = [im]
    else:
        res = []
    for n in range(2, max(ns)):
        im = _max_pool(im)
        if n in ns:
            res.append(im)
    return np.hstack([np.ravel(y) for y in res])


def group_by_label(samples, labels):
    """group samples by label"""

    unique_labels = sorted(list(set(labels)))

    grouped = [[x for x, y in zip(samples, labels) if y == label]
               for label in unique_labels]

    res_unsorted = list(zip(unique_labels, grouped))
    order = np.argsort([len(x) for x in grouped])
    return [res_unsorted[idx] for idx in order]


def balance(samples, labels, balance_factor, adjust_func):
    """create a balanced dataset by subsampling classes or generating new
    samples"""

    grouped = group_by_label(samples, labels)

    if balance_factor <= 1.0:
        target_group_size = int(len(grouped[-1][1]) * balance_factor)
    else:
        target_group_size = int(balance_factor)

    grouped_balanced = []
    for label, group in grouped:
        if len(group) > target_group_size:
            group_resized = random.sample(group, target_group_size)
        else:
            group_resized = [x for x in group]
            while len(group_resized) < target_group_size:
                group_resized.append(adjust_func(random.choice(group)))
        grouped_balanced.append((label, group_resized))

    pairs = [(y, x[0]) for x in grouped_balanced for y in x[1]]
    return zip(*pairs)


def transform_random(image):
    """apply a small random transformation to an image"""

    # TODO: make ranges of random numbers input parameters
    trans = np.random.rand(2) * 2 - 1.0
    rot = np.random.rand(4) * 0.32 - 0.16

    x_size = image.shape[1]
    y_size = image.shape[0]

    trans_to_center = np.float32(
        [[1, 0, -x_size / 2.0], [0, 1, -y_size / 2.0], [0, 0, 1]])
    trans_from_center = np.float32(
        [[1, 0, x_size / 2.0], [0, 1, y_size / 2.0], [0, 0, 1]])
    trans_random = np.float32(
        [[1 + rot[0], 0 + rot[1], trans[0]],
         [0 + rot[2], 1 + rot[3], trans[1]],
         [0, 0, 1]])

    tmat = np.dot(trans_from_center, np.dot(trans_random, trans_to_center))[0:2, :]
    image_new = cv2.warpAffine(
        image, tmat,
        (image.shape[1], image.shape[0]),
        borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", image_new)
    # cv2.waitKey()

    return image_new
