"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial

import cv2
import numpy as np

from handwriting import ml, improc, cnn

VISUALIZE = False

def build_current_best_process(
        data_train, labels_train):

    """build the current best character classification process"""

    start_row = 16
    pad_image_96 = partial(improc.pad_image, width=96, height=96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        image = image[start_row:, :]
        return 255.0 - improc.grayscale(improc.align(improc.filter_cc(pad_image_96(image))))
        # return 255.0 - improc.grayscale(improc.align(pad_image_96(image)))


    if True:

        def feat_extractor(image):
            """convert image to feature vector"""
            img_p = prep_image(image)
            img_g = img_p / 255.0

            # return ml.downsample_4(img_g)
            # return ml.downsample_multi(img_g, [1.0, 0.5])
            return improc.downsample_multi(img_g, [0.5, 0.25])

            # return np.hstack(
            #      (ml.column_ex(img_g), ml.column_ex(ml.max_pool(img_g))))

            # fft = np.fft.fft2(ml.max_pool(img_g), norm="ortho")
            # return np.hstack((
            #     np.ravel(np.absolute(fft)),
            #     # np.ravel(np.angle(fft)) / np.pi
            #     ))

            # grad_0, grad_1 = np.gradient(img_g)
            # return np.hstack((
            #     ml.max_pool_multi(grad_0, [2]),
            #     ml.max_pool_multi(grad_1, [2])))

            # return np.ravel(ml.max_pool_multi(img_g, [2]))

            # img_b = np.array(img_g * 255, dtype=np.uint8)
            # hog = cv2.HOGDescriptor(
            #     (96, 96), (8, 8), (8, 8), (2, 2), 8)
            # return np.ravel(hog.compute(img_b))

        if VISUALIZE:
            # visualize training data
            from handwriting.prediction import Sample
            from handwriting import charclass
            for cur_label, group in ml.group_by_label(data_train, labels_train):
                print("label:", cur_label)
                group_prepped = [(prep_image(x), None) for x in group]
                # print(np.min(group_prepped[0][0]), np.max(group_prepped[0][0]))
                group_pred = [Sample(x, cur_label, 0.0, False) for x in group_prepped]
                chars_working, chars_done = charclass.label_chars(group_pred)

        print("--extracting features from training data")
        feats_train = [feat_extractor(x) for x in data_train]
        print("--building feature selector")
        feat_selector = ml.feat_selection_pca(0.99)(feats_train) # 0.95
        # feat_selector = lambda x: x
        print("--selecting features from training data")
        feats_train = feat_selector(feats_train)

        print("--training classifier")

        # classifier, classifier_score = ml.train_classifier(
        #     # build_fit_model=ml.svc,
        #     build_fit_model=ml.linear_svc,
        #     score_func=ml.score_accuracy,
        #     n_splits=5,
        #     feats=feats_train,
        #     labels=labels_train,
        #     c=np.logspace(-2, 0, 10),
        #     # gamma=np.logspace(-5, 1, 8),
        #     # c=np.logspace(-2, 2, 7)
        # )

        classifier = ml.train_classifier(
            build_fit_model=ml.nn_classifier,
            score_func=ml.score_accuracy,
            cross_validation=ml.kfold_cross_validation(5), # ml.holdout_validation(0.2),
            feats=feats_train,
            labels=labels_train,
            hidden_layer_sizes=[(256, 128), (256, 64), (256, 32)],
            # alpha=[0.0001, 0.001, 0.01],
            alpha=[0.0001]
        )

    else:

        def feat_extractor(image):
            """convert image to feature vector"""
            img_p = prep_image(image)
            img_g = img_p / 255.0
            return img_g

        feats_train = [feat_extractor(x) for x in data_train]
        feat_selector = lambda x: x
        feats_train = feat_selector(feats_train)

        classifier = cnn.experimental_cnn(
            max_epochs=256,
            learning_rate=0.001,
            momentum=0.9,
            log_filename="log.txt"
        )(
            feats_train,
            labels_train
        )

    classify_char_image = ml.classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier)
