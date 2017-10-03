"""

Given annotations, segment images to extract lines, words, and letters.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import cv2
import numpy as np

import geom


VISUALIZE = False


def extract_line_image(line, image, px_above, px_below):

    """extract line annotation image slice"""

    diag = int(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))


    _, theta = geom.points_to_polar(*line)

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # angle is in degrees
    t_mat = cv2.getRotationMatrix2D(center, theta * 180.0 / np.pi, 1.0)
    rotated_image = cv2.warpAffine(image, t_mat, (diag, diag))

    line_full = geom.line_segment_within_image(line, image.shape)

    p1 = np.array(line_full[0:2], dtype=np.int)
    p2 = np.array(line_full[2:4], dtype=np.int)

    pts_t = cv2.transform(np.array([[p1, p2]]), t_mat)
    p1_t = pts_t[0][0]
    p2_t = pts_t[0][1]

    # segment by slicing a fixed amount above and below the line
    line_y = p1_t[1] # should be very close to p2_t[1] - could average the two
    full_line_image = rotated_image[(line_y - px_above):(line_y + px_below), :]

    # discard the extra at the left and right
    x_range = np.sort([p1_t[0], p2_t[0]])
    line_image = full_line_image[:, x_range[0]:x_range[1]]

    # print(rotated_image.shape)
    # cv2.namedWindow("rot", cv2.WINDOW_NORMAL)
    # cv2.imshow("rot", rotated_image)
    # cv2.resizeWindow("rot", int(rotated_image.shape[1] / 5), int(rotated_image.shape[0] / 5))

    if VISUALIZE:
        # draw_lines(rotated_image, [np.hstack((p1_t, p2_t)).tolist()])
        print(p1, p2, " -> ", p1_t, p2_t)
        print(full_line_image.shape)
        print(line_image.shape)
        cv2.namedWindow("line", cv2.WINDOW_NORMAL)
        cv2.imshow(
            "line", cv2.resize(line_image,
            (int(line_image.shape[1] / 2), int(line_image.shape[0] / 2))))
        cv2.waitKey(0)

    return line_image


def extract_letters(word_image, lgaps):

    """given a word image and line gap positions, extract letter images."""

    # Unlike word extraction, this is its own function because the logic here
    # is potentially more complex.

    letter_width_min = 2

    lgaps = [0] + lgaps + [word_image.shape[1]]

    letter_images = []
    for idx in range(len(lgaps) - 1):
        letter_image = np.copy(word_image[:, lgaps[idx]:lgaps[idx + 1], :])
        if letter_image.shape[1] > letter_width_min:
            letter_images.append(letter_image)

    return letter_images
