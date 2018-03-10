# -*- coding: utf-8 -*-
"""

Train a character classifier.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import gc
import random
import sys

import attr
import numpy as np
import sklearn
import torch

from handwriting import util, charclass, improc
from handwriting import ml, imml
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
    do_balance = attr.ib()
    balance_factor = attr.ib()
    do_align = attr.ib()
    batch_size = attr.ib()
    max_epochs = attr.ib()
    train_idxs = attr.ib()
    test_idxs = attr.ib()


CONFIG_DEFAULT = Config(
    pad_width=96,
    pad_height=96,
    start_row=0,
    do_balance=True,
    balance_factor=1024,
    do_align=True,
    batch_size=16,
    max_epochs=16,
    train_idxs=list(range(5, 15)),
    test_idxs=[15]
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

            return [0.0, 0.0, accuracy]

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
        model_filename = "models/classify_characters.pkl"
    else:
        model_filename = argv[3]

    print("run_charclassml")
    print("---------------")
    cf.pretty_print(config)
    print("mode:", mode)
    print("model filename:", model_filename)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    min_label_examples = 1
    do_destructive_prepare_balance = True

    train_filenames = data.pages(config.train_idxs)
    test_filenames = data.pages(config.test_idxs)

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = _load_samples(train_filenames)

    # eliminate groups from training and test
    # where we have less than a certain number of samples or they aren't
    # characters that we currently want to train on
    train_gr = dict(ml.group_by_label(
        data_train_unbalanced, labels_train_unbalanced))
    keep_labels = sorted(
        [x for x, y in train_gr.items()
         if len(y) >= min_label_examples and x not in IGNORE_CHARS])
    print("keep labels:", keep_labels)

    train_grf = {x: y for x, y in train_gr.items() if x in keep_labels}
    data_train_unbalanced, labels_train_unbalanced = zip(*[
        (y, x[0]) for x in train_grf.items() for y in x[1]])

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

    print("prepared training data size:", util.mbs(data_train_unbalanced), "MiB")
    print()

    if config.do_balance:
        # balance classes in training set
        pad_image = partial(
            improc.pad_image,
            width=config.pad_width,
            height=config.pad_height)
        data_train, labels_train = ml.balance(
            data_train_unbalanced,
            labels_train_unbalanced,
            config.balance_factor,
            pipe(
                pad_image,  # pad before rotations
                partial(
                    improc.transform_random,
                    trans_size=[0.0, 0.0],
                    rot_size=0.5,
                    scale_size=0.1)))
    else:
        data_train = data_train_unbalanced
        labels_train = labels_train_unbalanced

    # load test set
    data_test, labels_test = _load_samples(test_filenames)

    test_gr = dict(ml.group_by_label(data_test, labels_test))
    test_grf = {x: y for x, y in test_gr.items() if x in keep_labels}
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

    if mode == MODE_TRAIN:

        print("training model...")

        if True:
            # train a CNN

            # TODO: load separate validation data to use instead of test data
            data_validate = data_test
            labels_validate = labels_test

            prepare_callback = build_prepare_callback(data_validate, labels_validate)

            proc = imml.build_classification_process_cnn(
                data_train,
                labels_train,
                config.pad_width,
                config.pad_height,
                config.start_row,
                do_align=config.do_align,
                batch_size=config.batch_size,
                max_epochs=config.max_epochs,
                epoch_log_filename=model_filename + ".log.txt",
                prepare_callback=prepare_callback,
                save_model_filename=model_filename + ".wip")
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
