"""

Machine learning for character classification.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

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
        # return ml._align(ml._filter_cc(pad_image_96(image)))
        return ml._filter_cc(pad_image_96(image))

    def feat_extractor(image):
        """convert image to feature vector"""
        img_p = prep_image(image)
        img_g = ml.grayscale(img_p)
        # grad_0, grad_1 = np.gradient(img_g)
        # return np.hstack((
        #     ml._max_pool_multi(grad_0, [3, 4, 5]),
        #     ml._max_pool_multi(grad_1, [3, 4, 5])))
        return ml._downsample_4(img_g)

    feats_train = [feat_extractor(x) for x in data_train]
    feat_selector = ml.build_feat_selection_pca(feats_train, 0.90)
    feats_train = feat_selector(feats_train)

    classifier, classifier_score = ml.train_svc_classifier(
        feats_train, labels_train, 4, support_ratio_max)

    classify_char_image = ml.build_classification_process(
        feat_extractor, feat_selector, classifier)

    return (classify_char_image,
            prep_image, feat_extractor, feat_selector,
            classifier, classifier_score)
