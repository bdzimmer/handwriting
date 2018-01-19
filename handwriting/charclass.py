# -*- coding: utf-8 -*-
"""

More complex methods of manual character annotation.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import cv2
import numpy as np

from handwriting import improc, util, ml
from handwriting.prediction import Sample


def label_chars(chars, width=16, height=8, border=True):

    """label and categorize character images with the mouse and keyboard"""

    # TODO: get rid of default parameters
    # TODO: optionally show the word image (second element of data tuple)
    # assumes the first part of the data tuples are bitmaps already padded to
    # the same size

    if len(chars) == 0:
        return [], []

    chars_working = [(x, False) for x in chars]
    chars_done = []

    new_char = ["`"]
    idx = [0]
    type_mode = True

    patch_width = chars[0].data[0].shape[1]
    patch_height = chars[0].data[0].shape[0]

    if len(chars[0].data[0]) == 3:
        blank_patch = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)
    else:
        blank_patch = np.zeros((patch_height, patch_width), dtype=np.uint8)

    def on_mouse(event, mouse_x, mouse_y, flags, params):
        """helper"""
        if event == cv2.EVENT_LBUTTONDOWN:
            idx[0] = int(mouse_y / patch_height) * 16 + int(mouse_x / patch_width)
            if idx[0] < len(chars_working):
                pred = chars_working[idx[0]][0]
                pred_new = pred.copy(result=new_char[0], verified=True)
                chars_working[idx[0]] = (pred_new, True)
                draw()

    def draw():
        """helper"""
        print("total working characters:", len([x for x, y in chars_working if not y]))
        bmps = [np.copy(x.data[0]) if not y else blank_patch
                for x, y in chars_working]
        bmps_color = []
        for bmp in bmps:
            if len(bmp.shape) == 2:
                bmp = np.expand_dims(bmp, 2).repeat(3, 2)
            if border:
                cv2.rectangle(
                    bmp,
                    (0, 0),
                    (bmp.shape[1] - 1, bmp.shape[0] - 1),
                    (255, 0, 0))
                # debugging character positions
                cv2.rectangle(
                    bmp,
                    (8, 0),
                    (bmp.shape[1] - 9, bmp.shape[0] - 1),
                    (0, 255, 0))
                cv2.line(
                    bmp,
                    (bmp.shape[1] // 2, 0),
                    (bmp.shape[1] // 2, bmp.shape[0] - 1),
                    (0, 0, 255))
            bmps_color.append(bmp)
        patch_im = util.patch_image(bmps_color, width, height)
        cv2.imshow("characters", patch_im)

    cv2.namedWindow("characters", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("characters", on_mouse, 0)
    draw()

    while True:
        if len(chars_done) == 0 and len([x for x, y in chars_working if not y]) == 0:
            break
        key = cv2.waitKey(60000)
        if key == 27: # escape
            break
        elif key == 9: # tab
            print("toggling type mode")
            type_mode = not type_mode
            continue
        elif key == 13: # enter
            chars_done = chars_done + [x for x in chars_working if x[1]]
            chars_working = [x for x in chars_working if not x[1]]
            idx[0] = 0
            draw()
            continue
        elif key == 8 and type_mode: # backspace
            idx[0] = idx[0] - 1
            if idx[0] < 0:
                idx[0] = 0
            chars_working[idx[0]] = (chars_working[idx[0]][0], False)
            draw()
            continue
        elif key == 2555904: # right arrow
            for c_idx in range(width * height):
                if c_idx < len(chars_working):
                    chars_working[c_idx] = (chars_working[c_idx][0], True)
            chars_done = chars_done + [x for x in chars_working if x[1]]
            chars_working = [x for x in chars_working if not x[1]]
            idx[0] = 0
            draw()
            continue
        elif key == 32: # space
            new_char[0] = "`"
            print("marking invalid")
        else:
            new_char[0] = chr(key & 0xFF)
            print("marking " + new_char[0])

        if type_mode:
            print(idx[0])
            pred = chars_working[idx[0]][0]
            pred_new = pred.copy(result=new_char[0], verified=True)
            chars_working[idx[0]] = (pred_new, True)
            idx[0] = idx[0] + 1
            if idx[0] >= len(chars_working):
                idx[0] = len(chars_working) - 1
            draw()

    chars_done = chars_done + [x for x in chars_working if x[1]]
    chars_working = [x for x in chars_working if not x[1]]

    cv2.destroyWindow("characters")

    return [x[0] for x in chars_working], [x[0] for x in chars_done]


def visualize_training_data(data_train, labels_train, color_to_gray):
    """visualize training data"""

    for cur_label, group in ml.group_by_label(data_train, labels_train):
        print("label:", cur_label)
        group_prepped = [(color_to_gray(x), None) for x in group]
        group_pred = [Sample(x, cur_label, 0.0, False) for x in group_prepped]
        _ = label_chars(group_pred)


def main():
    """main program"""

    label_mode = False
    label_ml = True

    patch_width = 96
    patch_height = 96

    pad = lambda x: improc.pad_image(x, patch_width, patch_height)

    def pad_preds(preds):
        """helper"""
        return [p.copy(data=(pad(p.data[0]), p.data[1], p.data[0])) for p in preds]

    def unpad_preds(preds):
        """helper"""
        return [p.copy(data=(p.data[2], p.data[1])) for p in preds]

    input_filename = "20170929_3.png.character.pkl"
    with open(input_filename, "rb") as input_file:
        chars = pickle.load(input_file)

    print("total characters:", len(chars))

    if label_mode:

        chars_verified = [x for x in chars if x.verified]
        chars_working = [x for x in chars if not x.verified]

        print("verified:", len(chars_verified))
        print("working:", len(chars_working))

        if len(chars_working) > 0:

            chars_working, chars_done = label_chars(pad_preds(chars_working))
            res = chars_verified + unpad_preds(chars_working) + unpad_preds(chars_done)

            with open(input_filename, "wb") as output_file:
                pickle.dump(res, output_file)

    else:

        if label_ml:
            # TODO: do this only for unverified chars
            print("generating labels using ML")
            svc_predict, _ = util.load_dill("char_class_svc.pkl")
            labels = svc_predict([x.data for x in chars])
            chars = [x.copy(result=y) for x, y in zip(chars, labels)]

        unique_labels = sorted(list(set([x.result for x in chars if x.result is not None])))
        invalid_chars = [x for x in chars if x.result is None]
        preds_grouped = [[x for x in chars if x.result == cur_char] for cur_char in unique_labels]

        chars_confirmed = []
        chars_redo = []

        print("mark incorrectly labeled characters")
        for cur_label, group in zip(unique_labels, preds_grouped):
            print(cur_label)

            chars_working, chars_done = label_chars(pad_preds(group))
            chars_confirmed += unpad_preds(chars_working)
            chars_redo += unpad_preds(chars_done)

        if len(chars_redo) > 0:
            print("label these characters correctly")
            chars_redo = pad_preds([x.copy(verified=False) for x in chars_redo])
            chars_working, chars_done = label_chars(chars_redo)

            res = chars_confirmed + unpad_preds(chars_working) + unpad_preds(chars_done) + invalid_chars

            with open(input_filename, "wb") as output_file:
                pickle.dump(res, output_file)


if __name__ == "__main__":
    main()
