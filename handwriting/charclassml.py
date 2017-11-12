"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial

import numpy as np

from handwriting import ml


def build_current_best_process(
        data_train, labels_train, support_ratio_max):

    """build the current best character classification process"""

    pad_image_96 = lambda x: ml.pad_image(x, 96, 96)

    def prep_image(image):
        """prepare an image (result can still be visualized as an image)"""
        start_row = 16
        image = image[start_row:, :]
        return ml._align(ml._filter_cc(pad_image_96(image)))
        # return ml._filter_cc(pad_image_96(image))

    def feat_extractor(image):
        """convert image to feature vector"""
        img_p = prep_image(image)
        img_g = ml.grayscale(img_p)
        return ml._downsample_4(img_g)

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = ml.build_feat_selection_pca(feats_train, 0.90)
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = ml.train_classifier(
        fit_model=partial(
            ml.build_svc_fit,
            support_ratio_max=support_ratio_max),
        score_func=ml.score_accuracy,
        n_splits=4,
        feats=feats_train,
        labels=labels_train,
        c=np.logspace(1, 2, 4),
        gamma=np.logspace(-2, 0, 7))

    classify_char_image = ml.build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)
