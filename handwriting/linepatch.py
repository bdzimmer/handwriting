"""

Experimentation with line detection models.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import numpy as np
import cv2
import sklearn.svm

from handwriting import geom, util

def main():

    """main program"""

    input_file = "20170929_1.png"
    lines_filename = input_file + ".lines.pkl"

    im = cv2.imread(input_file)

    with open(lines_filename, "rb") as lines_file:
        preds = pickle.load(lines_file)
        lines = preds[0].result

    patch_size = 32
    n_patches = 20000
    patches = []

    hp = int(patch_size / 2)

    def min_dist_to_line(x, y):
        """helper"""
        return np.min([geom.point_to_line(x, y, *line) for line in lines])

    while len(patches) < n_patches:

        # choose a random location
        x = int(np.random.uniform(0, im.shape[1]))
        y = int(np.random.uniform(0, im.shape[0]))

        # keep it that are within a certain distance from a line
        dist = min_dist_to_line(x, y)

        if dist > 200:
            continue

        # extract patch bitmap
        bmp = im[(y - hp):(y + hp), (x - hp):(x + hp)]
        if bmp.shape[0] != patch_size or bmp.shape[1] != patch_size:
            continue

        # classify by distance to nearest ground-truth line
        is_line = dist < patch_size / 3
        patches.append((x, y, dist, is_line, bmp))

    # balance classes

    line_patches = [x for x in patches if x[3]]
    nonline_patches = [x for x in patches if not x[3]]

    patches = line_patches + nonline_patches[:len(line_patches)]

    print("patch count after balancing classes:", len(patches))

    disp_im = np.copy(im)
    for patch in patches:
        x, y, dist, is_line, bmp = patch
        color = (0, 255, 0) if is_line else (255, 0, 0)
        cv2.rectangle(disp_im, (x - hp, y - hp), (x + hp, y + hp), color, 4)

    def get_feat(x):
        """helper"""
        res = np.hstack((
            np.histogram(x[:, :, 0], bins=12, range=(0, 255))[0],
            np.histogram(x[:, :, 1], bins=12, range=(0, 255))[0],
            np.histogram(x[:, :, 2], bins=12, range=(0, 255))[0]))
        return res / (patch_size * patch_size)

    feats = np.array([get_feat(x[4]) for x in patches])
    labels = np.array([(1 if x[3] else 0) for x in patches])

    # fit an svm
    svc = sklearn.svm.SVC(probability=True)
    svc.fit(feats, labels)
    print(svc.n_support_)
    print(svc.score(feats, labels))

    # predict in a new floating point image
    scale = 8
    pred_image = np.zeros((int(im.shape[0] / scale), int(im.shape[1] / scale)))
    for y in range(0, im.shape[0], scale):
        print(y, "/", im.shape[0])
        for x in range(0, im.shape[1], scale):
            bmp = im[(y - hp):(y + hp), (x - hp):(x + hp)]
            if bmp.shape[0] != patch_size or bmp.shape[1] != patch_size:
                continue
            feat = get_feat(bmp).reshape(1, -1)
            res = svc.predict_proba(feat)
            pred_image[int(y / scale), int(x / scale)] = res[0, 1]

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", disp_im)
    cv2.resizeWindow("image", int(im.shape[1] / 2), int(im.shape[0] / 2))

    cv2.namedWindow("line patches", cv2.WINDOW_NORMAL)
    cv2.imshow("line patches", util.patch_image([x[4] for x in patches if x[3]]))

    cv2.namedWindow("non-line patches", cv2.WINDOW_NORMAL)
    cv2.imshow("non-line patches", util.patch_image([x[4] for x in patches if not x[3]]))

    cv2.namedWindow("prediction", cv2.WINDOW_NORMAL)
    cv2.imshow("prediction", pred_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
