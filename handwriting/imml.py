"""

Processes for training machine learning models for images.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

from functools import partial
import gc
import os

import numpy as np

from handwriting import util, charclass, ml, func, improc, cnn

VISUALIZE = False

DO_LAZY_EXTRACTION = True
CALLBACK_RATE = 1


def build_classification_process_cnn(
        data_train,
        labels_train,
        pad_width,
        pad_height,
        start_row,
        do_align,
        batch_size,
        max_epochs,
        epoch_log_filename,
        prepare_callback,
        save_model_filename):

    """build a classification process for images using a CNN"""

    # TODO: do I pull out all the feature extractors or flatten the configuration?
    # I'm more inclined to flatten the configuration for this approach.

    pad_only = partial(improc.pad_image, width=pad_width, height=pad_height)

    if do_align:
        pad_image = lambda x: improc.align(pad_only(x))
    else:
        pad_image = pad_only

    def color_to_grayuint(image):
        """prepare image to uint8"""
        image = image[start_row:, :]
        return np.array(
            255.0 - improc.grayscale(pad_image(image)),
            dtype=np.uint8)

    def grayuint_to_grayfloat(image):
        """convert uint8 image to floating point"""
        img_g = image / 255.0 - 0.5
        return np.array(img_g, np.float32)

    if VISUALIZE:
        charclass.visualize_training_data(data_train, labels_train, color_to_grayuint)

    feat_extractor = func.pipe(color_to_grayuint, grayuint_to_grayfloat)

    print("preparing callback...", end="", flush=True)
    callback = prepare_callback(feat_extractor)
    print("done")

    if not DO_LAZY_EXTRACTION:
        feats_train = [feat_extractor(x) for x in data_train]
        lazy_extractor = None
    else:
        feats_train = [color_to_grayuint(x) for x in data_train]
        del data_train
        gc.collect()
        lazy_extractor = grayuint_to_grayfloat

    print(
        "training features (input to CNN) size:",
        util.mbs(feats_train), "MiB")

    if prepare_callback is not None:
        filename, ext = os.path.splitext(epoch_log_filename)
        callback_log_filename = filename + "_callback" + ext
    else:
        callback_log_filename = None

    classifier = cnn.experimental_cnn(
        batch_size=batch_size,  # 16
        max_epochs=max_epochs,
        learning_rate=0.001,
        momentum=0.9,
        epoch_log_filename=epoch_log_filename, # "log_charpos.txt",
        callback_log_filename=callback_log_filename,
        callback=callback,
        callback_rate=CALLBACK_RATE,
        lazy_extractor=lazy_extractor,
        save_model_filename=save_model_filename
    )(
        feats_train,
        labels_train
    )

    classify_image = lambda image: classifier(feat_extractor(image))

    return (
        classify_image,
        color_to_grayuint,
        feat_extractor,
        classifier)


# These next two functions are around the best traditional ML processes I came up with
# for the character position and character classification problems.

def build_classification_process_charpos(
        data_train,
        labels_train,
        pad_width,
        pad_height,
        start_row):

    """build a classification process using traditional machine learning
    with cross validation"""

    pad_image = partial(improc.pad_image, width=pad_width, height=pad_height)

    def color_to_grayfloat(image):
        """prepare an image (result is still a 2d image)"""
        image = image[start_row:, :]
        return 255.0 - improc.grayscale(pad_image(image))

    def img_to_vec(image):
        """convert image to 1D feature vector"""
        img_p = color_to_grayfloat(image)
        img_g = img_p / 255.0
        img_g = img_g / np.max(img_g)

        return improc.max_pool_multi(img_g, [2])

        grad_0, grad_1 = np.gradient(img_g)
        return np.hstack((
            improc.max_pool_multi(grad_0, [2]),
            improc.max_pool_multi(grad_1, [2]),
            improc.max_pool_multi(img_g, [2])))

    if VISUALIZE:
        charclass.visualize_training_data(
            data_train, labels_train, color_to_grayfloat)

    feats_train_intermediate = [img_to_vec(x) for x in data_train]
    feat_selector = ml.feat_selection_pca(0.99)(feats_train_intermediate)
    # feat_selector = ml.feat_scaler(True)(feats_train_intermediate)
    # feat_selector = lambda x: x

    feat_extractor = lambda image: feat_selector(img_to_vec(image))

    feats_train = [feat_extractor(x) for x in data_train]
    print("feature length:", len(feats_train[0]))

    classifier = ml.train_classifier(
        build_fit_model=ml.linear_svc,
        cross_validation=ml.kfold_cross_validation(5),
        score_func=partial(ml.score_auc, decision_function=True),
        feats=feats_train,
        labels=labels_train,
        c=np.logspace(-3, 0, 10),
    )

    # classifier = ml.train_classifier(
    #     build_fit_model=ml.nn_classifier,
    #     cross_validation=ml.kfold_cross_validation(10),
    #     score_func=partial(ml.score_auc, decision_function=False),
    #     feats=feats_train,
    #     labels=labels_train,
    #     # hidden_layer_sizes=[(16, 16), (32, 32), (256, 128), (256, 64), (256, 32)],
    #     # hidden_layer_sizes=[(128, 128, 128), (256, 256, 256)],
    #     # hidden_layer_sizes=[(128,), (256,), (128, 128), (256, 256)],
    #     hidden_layer_sizes=[(128, 128, 128), (128, 128, 128, 128), (64, 64), (64, 64, 64), (64, 64, 64, 64)],
    #     # alpha=[0.0001, 0.01]
    #     alpha=[0.0001, 0.001, 0.01]
    # )

    classify_charpos_image = lambda image: classifier(feat_extractor(image))

    return (
        classify_charpos_image,
        color_to_grayfloat,
        feat_extractor,
        classifier)


def build_classification_process_charclass(
        data_train,
        labels_train,
        pad_width,
        pad_height,
        start_row):

    """build classification process using traditional machine learning
    with cross validation"""

    pad_image = partial(improc.pad_image, width=pad_width, height=pad_height)

    def color_to_grayfloat(image):
        """prepare an image (result is still a 2d image)"""
        image = image[start_row:, :]
        # return 255.0 - improc.grayscale(improc.align(improc.filter_cc(pad_image(image))))
        return 255.0 - improc.grayscale(improc.align(pad_image(image)))

    def img_to_vec(image):
        """convert image to feature vector"""
        img_p = color_to_grayfloat(image)
        img_g = img_p / 255.0
        # return ml.downsample_4(img_g)
        # return ml.downsample_multi(img_g, [1.0, 0.5])
        return improc.downsample_multi(img_g, [0.5, 0.25])

    if VISUALIZE:
        charclass.visualize_training_data(
            data_train, labels_train, color_to_grayfloat)

    print("--extracting features from training data")
    feats_train_intermediate = [img_to_vec(x) for x in data_train]
    print("--building feature selector")
    feat_selector = ml.feat_selection_pca(0.99)(feats_train_intermediate) # 0.95
    # feat_selector = lambda x: x
    print("--selecting features from training data")

    feat_extractor = lambda image: feat_selector(img_to_vec(image))

    feats_train = [feat_extractor(x) for x in data_train]

    print("--training classifier")

    classifier = ml.train_classifier(
        # build_fit_model=ml.svc,
        build_fit_model=ml.linear_svc,
        score_func=ml.score_accuracy,
        cross_validation=ml.kfold_cross_validation(5),
        feats=feats_train,
        labels=labels_train,
        c=np.logspace(-2, 0, 10),
        # gamma=np.logspace(-5, 1, 8),
        # c=np.logspace(-2, 2, 7)
    )

    # classifier = ml.train_classifier(
    #     build_fit_model=ml.nn_classifier,
    #     score_func=ml.score_accuracy,
    #     cross_validation=ml.kfold_cross_validation(5), # ml.holdout_validation(0.2),
    #     feats=feats_train,
    #     labels=labels_train,
    #     hidden_layer_sizes=[(256, 128), (256, 64), (256, 32)],
    #     # alpha=[0.0001, 0.001, 0.01],
    #     alpha=[0.0001]
    # )

    classify_char_image = lambda image: classifier(feat_extractor(image))

    return (
        classify_char_image,
        color_to_grayfloat,
        feat_extractor,
        classifier)
