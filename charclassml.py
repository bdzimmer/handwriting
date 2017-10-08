"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import random

import cv2
import numpy as np
from sklearn.svm import SVC

import charclass
import util

from prediction import Prediction


def _fit_svc(feats, labels):
    """fit an svm classifier"""

    # TODO: feature selection; hyperparameter tuning with cross validation

    svc = SVC(probability=True, verbose=False)
    svc.fit(feats, labels)

    return svc


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
    second_largest_idx = np.argsort(sizes)[-2]

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
    """downsample a patch image and unravel to create a feature vector"""

    feats = np.ravel(
        cv2.resize(
            np.sum(image, axis=2) / (255.0 * 3.0),
            (int(image.shape[0] * scale_factor),
             int(image.shape[1] * scale_factor))))
    return feats


def _group_by_label(samples, labels):
    """group samples by label"""

    unique_labels = sorted(list(set(labels)))

    grouped = [[x for x, y in zip(samples, labels) if y == label]
               for label in unique_labels]

    res_unsorted = list(zip(unique_labels, grouped))
    order = np.argsort([len(x) for x in grouped])
    return [res_unsorted[idx] for idx in order]


def _balance(samples, labels, balance_factor, adjust_func):
    """create a balanced dataset by subsampling classes or generating new
    samples"""

    grouped = _group_by_label(samples, labels)

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


def _transform_random(image):
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
    image_new = cv2.warpAffine(image, tmat, image.shape[0:2], borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", new_image)
    # cv2.waitKey()

    return image_new


def main():
    """main program"""

    patch_width = 96
    patch_height = 96

    pad = lambda x: charclass.pad_char_bmp(x, patch_width, patch_height)

    train_filename = "20170929_2.png.character.pkl"
    test_filename = "20170929_1.png.character.pkl"

    data_to_feats = lambda x: _downsample(x[0], 0.4)

    def load_samples(filename):
        """load  valid character samples from file, converting the data field
        to a padded bitmap"""
        samples_valid = [x for x in util.load(filename)
                         if x.result is not None and x.verified]
        labels = [x.result for x in samples_valid]
        data_padded = [(_align(_filter_cc(pad(x.data[0]))), x.data[1], x.data[0])
                       for x in samples_valid]
        return data_padded, labels

    print("loading and balancing datasets")

    def perturb_func(data):
        """helper"""
        im_new = _transform_random(data[0])
        return (im_new, data[1], data[2])

    data_train_unbalanced, labels_train_unbalanced = load_samples(train_filename)
    data_train, labels_train = _balance(
        data_train_unbalanced, labels_train_unbalanced, 0.5, perturb_func)

    feats_train = [data_to_feats(x) for x in data_train]

    data_test, labels_test = load_samples(test_filename)
    feats_test = [data_to_feats(x) for x in data_test]

    print("done")

    print("training size:", len(feats_train))
    print("test size:", len(feats_test))

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in _group_by_label(feats_train, labels_train)])

    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in _group_by_label(feats_test, labels_test)])

    print("fitting model")

    svc = _fit_svc(feats_train, labels_train)

    print("done")

    print("support vector counts:", svc.n_support_)
    print("score on training dataset", svc.score(feats_train, labels_train))
    print("score on test dataset", svc.score(feats_test, labels_test))

    # TODO: visualize ROC curves and confusion matrix

    labels_test_pred = svc.predict(feats_test)

    chars_confirmed = []
    chars_redo = []

    # show results
    for cur_label, group in _group_by_label(data_test, labels_test_pred):
        print(cur_label)
        group_pred = [Prediction(x, cur_label, 0.0, False) for x in group]
        chars_working, chars_done = charclass.label_chars(group_pred)
        chars_confirmed += chars_working
        chars_redo += chars_done


if __name__ == "__main__":
    main()
