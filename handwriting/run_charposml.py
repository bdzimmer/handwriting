# -*- coding: utf-8 -*-
"""

Train a classifier to classify vertical slices of a word image as within a
a letter or between letters.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from handwriting import charclassml as cml, util, charclass
from handwriting.prediction import Sample

VISUALIZE = False


def build_process(data_train, labels_train):

    """build a classification process"""

    # A lot of this is reused from the character classification process

    pad_image_96 = lambda x: cml.pad_image(x, 96, 96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        start_row = 16
        image = image[start_row:, :]
        # return cml._align(cml._filter_cc(pad_image_96(image)))
        # return cml._align(pad_image_96(image))
        return pad_image_96(image)

    def feat_extractor(image):
        """convert image to feature vector"""
        p_img = prep_image(image)
        return cml._downsample_4(p_img)

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = cml.build_feat_selection_pca(feats_train, 0.999)
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = cml.train_char_class_svc(
        feats_train, labels_train, 4)

    classify_char_image = cml.build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)


def main():
    """main program"""

    train_filenames = [
        "data/20170929_2.png.sample.pkl.8",
        "data/20170929_3.png.sample.pkl.5"]
    test_filenames = [
        "data/20170929_1.png.sample.pkl.14"]

    def load_samples(filenames):
        """helper"""

        # load multiple files
        line_poss = [y for x in filenames for y in util.load(x).result]

        # get word images and verified character positions
        word_ims = [word_pos.result
                    for line_pos in line_poss
                    for word_pos in line_pos.result.result
                    if word_pos.result.verified]

        # for each word image, extract images of the start character positions
        # and positions in the middle of characters
        extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]
        half_width = 4
        extract_char_half_width = lambda x, im: extract_char((x - half_width, x + half_width), im)

        char_start_ims = [extract_char_half_width(x.data[0], word_im.data)
                          for word_im in word_ims
                          for x in word_im.result]
        char_start_ims = [x for x in char_start_ims if x.shape[1] > 0]

        char_middle_ims = [extract_char_half_width(
                               int(x.data[0] + (x.data[1] - x.data[0]) * 0.5), word_im.data)
                           for word_im in word_ims
                           for x in word_im.result]
        char_middle_ims = [x for x in char_middle_ims if x.shape[1] > 0]

        # import cv2
        # for im in char_middle_ims:
        #     print(im)
        #     cv2.imshow("test", im)
        #     cv2.waitKey()

        data = char_start_ims + char_middle_ims
        labels = [True] * len(char_start_ims) + [False] * len(char_middle_ims)

        return data, labels

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = load_samples(train_filenames)

    # balance classes in training set
    balance_factor = 0.25
    data_train, labels_train = cml.balance(
        data_train_unbalanced, labels_train_unbalanced,
        balance_factor, cml.transform_random)

    # load test set
    data_test, labels_test = load_samples(test_filenames)

    test_gr = dict(cml.group_by_label(data_test, labels_test))
    test_grf = test_gr
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

    proc = build_process(
        data_train, labels_train)

    (classify_char_pos,
     prep_image, feat_extractor, feat_selector,
     classifier, classifier_score) = proc

    print("done")

    feats_test = feat_selector([feat_extractor(x) for x in data_test])
    print("score on test dataset", classifier_score(feats_test, labels_test))
    # TODO: visualize ROC curves and confusion matrix
    util.save_dill(proc, "models/classify_charpos.pkl")

    if VISUALIZE:
        labels_test_pred = classify_char_pos(data_test)
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
