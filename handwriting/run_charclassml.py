# -*- coding: utf-8 -*-
"""

Train a character classifier.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import sys

import attr
import numpy as np
import sklearn
import torch

from handwriting import util, charclass, improc
from handwriting import ml, imml, dataset
from handwriting import data, config as cf
from handwriting.func import pipe
from handwriting.prediction import Sample

VISUALIZE = False

MODE_TRAIN = "train"
MODE_TUNE = "tune"

IGNORE_CHARS = ["~"]


@attr.s
class Config:
    pad_width = attr.ib()
    pad_height = attr.ib()
    start_row = attr.ib()
    do_align = attr.ib()
    nn_arch = attr.ib()
    nn_opt = attr.ib()
    trans_x_size = attr.ib()
    trans_y_size = attr.ib()
    rot_size = attr.ib()
    scale_size = attr.ib()
    train: dataset.PrepConfig = attr.ib()
    dev: dataset.PrepConfig = attr.ib()
    test: dataset.PrepConfig = attr.ib()


CONFIG_DEFAULT = Config(
    pad_width=96,
    pad_height=96,
    start_row=0,
    do_align=True,
    nn_arch={},
    nn_opt={},
    trans_x_size=0.0,
    trans_y_size=0.0,
    rot_size=0.0,
    scale_size=0.0,
    train=dataset.PrepConfig(),
    dev=dataset.PrepConfig(),
    test=dataset.PrepConfig()
)


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


def build_prepare_callback(data_validate, labels_validate):

    def prepare_callback(feat_extractor):
        """given feature extractor, build callback to test the
        network during training"""
        feats_validate = [feat_extractor(x) for x in data_validate]

        print("validation features size:", util.mbs(feats_validate), "MiB")

        def callback(classifier):
            """helper"""
            print("predicting...", end="", flush=True)
            labels_validate_pred = [classifier(x) for x in feats_validate]
            print("done")

            # TODO: something generic here instead of sklearn
            accuracy = sklearn.metrics.accuracy_score(
                labels_validate, labels_validate_pred)
            print("validation accuracy:", accuracy)

            return [
                ("nothing", 0.0),
                ("nothing", 0.0),
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
        config.train = dataset.PrepConfig(**config.train)
        config.dev = dataset.PrepConfig(**config.dev)
        config.test = dataset.PrepConfig(**config.test)

    if len(argv) < 4:
        model_filename = "models/classify_characters.pkl"
    else:
        model_filename = argv[3]

    print("run_charclassml")
    print("---------------")
    cf.pretty_print(config)
    print("mode:", mode)
    print("model filename:", model_filename)

    torch.manual_seed(0)

    pad_image = partial(
        improc.pad_image,
        width=config.pad_width,
        height=config.pad_height)

    augment_func = pipe(
        pad_image,  # pad before rotations
        partial(
            improc.transform_random,
            trans_size=[config.trans_x_size, config.trans_y_size],
            rot_size=config.rot_size,
            scale_size=config.scale_size))

    filenames_train = data.pages(config.train.idxs)
    filenames_dev = data.pages(config.dev.idxs)
    filenames_test = data.pages(config.test.idxs)

    # for integration testing
    # filenames_train = data.pages([5, 6, 7])
    # filenames_dev = data.pages([8])
    # filenames_test = data.pages([9])

    print("loading and preparing datasets...")

    print("train files:", filenames_train)
    data_train_raw, labels_train_raw = _load_samples(filenames_train)
    data_train, labels_train = dataset.prepare(
        data_train_raw,
        labels_train_raw,
        config.train.do_subsample,
        config.train.subsample_size,
        config.train.do_prep_balance,
        config.train.do_balance,
        config.train.balance_size,
        config.train.do_augment,
        config.train.augment_size,
        augment_func)

    print("dev files:", filenames_dev)
    data_dev_raw, labels_dev_raw = _load_samples(filenames_dev)
    data_dev, labels_dev = dataset.prepare(
        data_dev_raw,
        labels_dev_raw,
        config.dev.do_subsample,
        config.dev.subsample_size,
        config.dev.do_prep_balance,
        config.dev.do_balance,
        config.dev.balance_size,
        config.dev.do_augment,
        config.dev.augment_size,
        augment_func)

    print("test files:", filenames_test)
    data_test_raw, labels_test_raw = _load_samples(filenames_test)
    data_test, labels_test = dataset.prepare(
        data_test_raw,
        labels_test_raw,
        config.test.do_subsample,
        config.test.subsample_size,
        config.test.do_prep_balance,
        config.test.do_balance,
        config.test.balance_size,
        config.test.do_augment,
        config.test.augment_size,
        augment_func)

    # filter by label

    min_label_examples = 1
    keep_labels = sorted(
        [x for x, y in ml.group_by_label(data_train, labels_train)
         if len(y) >= min_label_examples and x not in IGNORE_CHARS])
    data_train, labels_train = dataset.filter_labels(
        data_train, labels_train, keep_labels)
    data_dev, labels_dev = dataset.filter_labels(
        data_dev, labels_dev, keep_labels)
    data_test, labels_test = dataset.filter_labels(
        data_test, labels_test, keep_labels)

    print("done")

    print("train data size:", util.mbs(data_train), "MiB")
    print("dev data size:  ", util.mbs(data_dev), "MiB")
    print("test data size: ", util.mbs(data_test), "MiB")
    print("train count:    ", len(data_train))
    print("dev count:      ", len(data_dev))
    print("test count:     ", len(data_test))
    print()

    counts_train = ml.label_counts(labels_train)
    print("train group sizes:", counts_train[0])
    print()
    counts_dev = ml.label_counts(labels_dev)
    print("dev group sizes:", counts_dev[0])
    print()
    counts_test = ml.label_counts(labels_test)
    print("test group sizes:", counts_test[0])
    print()

    # print("training group sizes change in balancing:")
    # for x, y in train_unbalanced_counts[0]:
    #     count = train_counts[1].get(x, 0)
    #     print(x, round(count / y, 3))
    # print()

    if mode == MODE_TRAIN:

        print("training model...")

        if True:
            # train a CNN

            prepare_callback = build_prepare_callback(
                data_dev,
                labels_dev)

            proc = imml.build_classification_process_cnn(
                data_train,
                labels_train,
                config.pad_width,
                config.pad_height,
                config.start_row,
                do_align=config.do_align,
                nn_arch=config.nn_arch,
                nn_opt=config.nn_opt,
                epoch_log_filename=model_filename + ".log.txt",
                prepare_callback=prepare_callback,
                save_model_filename=model_filename + ".wip",
                tsv_filename=model_filename + ".status")
        else:
            # traditional ML
            proc = imml.build_classification_process_charclass(
                data_train,
                labels_train,
                config.pad_width,
                config.pad_height,
                config.start_row)

        classify_char_image, prep_image, feat_extractor, classifier = proc

        print("done")

        # summarize results

        labels_test_pred = [classify_char_image(x) for x in data_test]
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

        util.save_dill(proc, model_filename)

        if VISUALIZE:
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

        classify_char_image = util.load_dill(model_filename)[0]

        # evaluate score by label
        for label in keep_labels:

            keep_idxs = [idx for idx, lbl in enumerate(labels_test)
                         if lbl == label]

            data_test_subset = [data_test[idx] for idx in keep_idxs]
            labels_test_subset = [labels_test[idx] for idx in keep_idxs]
            labels_test_pred_subset = [classify_char_image(x)
                                       for x in data_test_subset]

            preds_grouped_counts = ml.group_by_label(
                data_test_subset, labels_test_pred_subset)

            # print(labels_test_pred_subset)

            score = sklearn.metrics.accuracy_score(
                labels_test_subset, labels_test_pred_subset)
            print(
                label, "\t", np.round(score, 3), "\t", len(keep_idxs), "\t",
                [(x[0], len(x[1])) for x in reversed(preds_grouped_counts)])


if __name__ == "__main__":
    main(sys.argv)
