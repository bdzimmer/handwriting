# -*- coding: utf-8 -*-
"""

Train a character classifier.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


from handwriting import charclassml as cml, util, charclass
from handwriting.prediction import Sample

VISUALIZE = False


def main():
    """main program"""

    train_filenames = [
        "data/20170929_2.png.character.pkl",
        "data/20170929_3.png.character.pkl"]
    test_filenames = [
        "data/20170929_1.png.character.pkl"]

    def load_samples(filenames):
        """load  valid character samples from file, converting the data field
        to a padded bitmap"""
        samples = [y for x in filenames for y in util.load(x)]
        samples_valid = [x for x in samples
                         if x.result is not None and x.verified]
        samples_valid = [(x.copy(result="`") if x.result is None else x)
                         for x in samples]
        data = [x.data[0] for x in samples_valid]
        labels = [x.result for x in samples_valid]
        return data, labels

    print("loading and balancing datasets...")

    # load training set
    data_train_unbalanced, labels_train_unbalanced = load_samples(train_filenames)

    # eliminate groups from training and test
    # where we have less than a certain number of samples or they aren't
    # characters that we currently want to train on
    min_label_examples = 1
    remove_labels = ["\"", "!", "/"]

    train_gr = dict(cml.group_by_label(
        data_train_unbalanced, labels_train_unbalanced))
    keep_labels = [x for x, y in train_gr.items()
                   if len(y) >= min_label_examples and x not in remove_labels]
    print("keep labels:", sorted(keep_labels))

    train_grf = {x: y for x, y in train_gr.items() if x in keep_labels}
    data_train_unbalanced, labels_train_unbalanced = zip(*[
        (y, x[0]) for x in train_grf.items() for y in x[1]])

    # balance classes in training set
    balance_factor = 66
    # balance_factor = 150
    data_train, labels_train = cml.balance(
        data_train_unbalanced, labels_train_unbalanced,
        balance_factor, cml.transform_random)

    # load test set
    data_test, labels_test = load_samples(test_filenames)

    test_gr = dict(cml.group_by_label(data_test, labels_test))
    test_grf = {x: y for x, y in test_gr.items() if x in keep_labels}
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

    proc = cml.build_current_best_process(
        data_train, labels_train)

    (classify_char_image,
     prep_image, feat_extractor, feat_selector,
     classifier, classifier_score) = proc

    print("done")

    feats_test = feat_selector([feat_extractor(x) for x in data_test])
    print("score on test dataset", classifier_score(feats_test, labels_test))
    # TODO: visualize ROC curves and confusion matrix
    util.save_dill(proc, "models/classify_characters.pkl")

    if VISUALIZE:
        labels_test_pred = classify_char_image(data_test)
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
