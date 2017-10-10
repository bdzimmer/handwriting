"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import random

import cv2
import numpy as np
from sklearn.svm import SVC

from handwriting import charclass, util
from handwriting.prediction import Prediction


def train_char_class_svc(data_train, labels_train, data_to_feats):
    """train an SVC character classifier"""

    feats_train = [data_to_feats(x) for x in data_train]

    # TODO: feature selection; hyperparameter tuning with cross validation

    svc = SVC(probability=True, verbose=False)
    svc.fit(feats_train, labels_train)

    print("support vector counts:", svc.n_support_)
    print("score on training dataset", svc.score(feats_train, labels_train))

    def predict(data_test):
        """helper"""
        feats_test = [data_to_feats(x) for x in data_test]
        return svc.predict(feats_test)

    # TODO: write a general score function in the future
    def score(data_test, labels_test):
        """helper"""
        feats_test = [data_to_feats(x) for x in data_test]
        return svc.score(feats_test, labels_test)

    return predict, score


def pad_char_bmp(char_bmp, width, height):
    """pad char bitmap in a larger bitmap"""

    start_row = 16

    char_bmp = char_bmp[start_row:, :]

    new_bmp = np.ones((height, width, 3), dtype=np.uint8) * 255

    xoff = int((width - char_bmp.shape[1]) / 2)
    yoff = int((height - char_bmp.shape[0]) / 2)

    new_bmp[yoff:(yoff + char_bmp.shape[0]), xoff:(xoff + char_bmp.shape[1])] = char_bmp

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
    image_new = cv2.warpAffine(
        image, tmat,
        (image.shape[1], image.shape[0]),
        borderValue=(255, 255, 255))

    # cv2.imshow("image", image)
    # cv2.imshow("new_image", image_new)
    # cv2.waitKey()

    return image_new


def main():
    """main program"""

    patch_width = 96
    patch_height = 96

    pad_image = lambda x: pad_char_bmp(x, patch_width, patch_height)
    prep_image = lambda x: _align(_filter_cc(pad_image(x)))
    data_to_feats = lambda x: _downsample(prep_image(x[0]), 0.3) # 0.4

    train_filenames = ["20170929_2.png.character.pkl", "20170929_3.png.character.pkl"]
    test_filenames = ["20170929_1.png.character.pkl"]

    def load_samples(filenames):
        """load  valid character samples from file, converting the data field
        to a padded bitmap"""
        samples = [y for x in filenames for y in util.load(x)]
        samples_valid = [x for x in samples
                         if x.result is not None and x.verified]
        labels = [x.result for x in samples_valid]
        data = [x.data for x in samples_valid]
        return data, labels

    print("loading and balancing datasets")

    def perturb_func(data):
        """helper"""
        im_new = _transform_random(data[0])
        return (im_new, data[1])

    # load training set

    data_train_unbalanced, labels_train_unbalanced = load_samples(train_filenames)

    # eliminate groups from training and test
    # where we have less than a certain number of samples

    train_gr = dict(_group_by_label(
        data_train_unbalanced, labels_train_unbalanced))
    keep_labels = [x for x, y in train_gr.items() if len(y) >= 2]
    print("keep labels:", sorted(keep_labels))

    train_grf = {x: y for x, y in train_gr.items() if x in keep_labels}
    data_train_unbalanced, labels_train_unbalanced = zip(*[
        (y, x[0]) for x in train_grf.items() for y in x[1]])

    # balance classes in training set

    data_train, labels_train = _balance(
        data_train_unbalanced, labels_train_unbalanced, 0.5, perturb_func)

    # load test set

    data_test, labels_test = load_samples(test_filenames)

    test_gr = dict(_group_by_label(data_test, labels_test))
    test_grf = {x: y for x, y in test_gr.items() if x in keep_labels}
    data_test, labels_test = zip(*[
        (y, x[0]) for x in test_grf.items() for y in x[1]])

    print("done")

    print("training size:", len(data_train))
    print("test size:", len(data_test))

    print(
        "training group sizes:",
        [(x[0], len(x[1]))
         for x in _group_by_label(data_train, labels_train)])

    print(
        "test group sizes:",
        [(x[0], len(x[1]))
         for x in _group_by_label(data_test, labels_test)])

    print("fitting model")

    svc_predict, svc_score = train_char_class_svc(
        data_train, labels_train, data_to_feats)

    print("done")

    print("score on test dataset", svc_score(data_test, labels_test))

    # TODO: visualize ROC curves and confusion matrix

    labels_test_pred = svc_predict(data_test)

    util.save_dill((svc_predict, svc_score), "char_class_svc.pkl")

    chars_confirmed = []
    chars_redo = []

    # show results
    for cur_label, group in _group_by_label(data_test, labels_test_pred):
        print(cur_label)
        group_prepped = [(prep_image(x[0]), x[1]) for x in group]
        group_pred = [Prediction(x, cur_label, 0.0, False) for x in group_prepped]
        chars_working, chars_done = charclass.label_chars(group_pred)
        chars_confirmed += chars_working
        chars_redo += chars_done


if __name__ == "__main__":
    main()