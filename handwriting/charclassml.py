"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial

import cv2
import numpy as np

from handwriting import ml

VISUALIZE = False

def build_current_best_process(
        data_train, labels_train, support_ratio_max):

    """build the current best character classification process"""

    pad_image_96 = lambda x: ml.pad_image(x, 96, 96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        start_row = 16
        image = image[start_row:, :]
        return 255.0 - ml.grayscale(ml._align(ml._filter_cc(pad_image_96(image))))

    def feat_extractor(image):
        """convert image to feature vector"""
        img_p = prep_image(image)
        img_g = img_p / 255.0

        # return ml._downsample_4(img_g)

        # grad_0, grad_1 = np.gradient(img_g)
        # return np.hstack((
        #     ml._max_pool_multi(grad_0, [2]),
        #     ml._max_pool_multi(grad_1, [2])))

        return np.ravel(ml._max_pool_multi(img_g, [2, 3]))

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

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = ml.build_feat_selection_pca(feats_train, 0.95)
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = ml.train_classifier(
        fit_model=partial(
            ml.build_svc_fit,
            support_ratio_max=support_ratio_max),
        score_func=ml.score_accuracy,
        n_splits=4,
        feats=feats_train,
        labels=labels_train,
        gamma=np.logspace(-5, 1, 16),
        c=np.logspace(-2, 2, 7))

    classify_char_image = ml.build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)
