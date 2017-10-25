# -*- coding: utf-8 -*-
"""

More complex methods of manual character annotation.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import cv2
import numpy as np

from handwriting import charclassml, util


def label_chars(chars):

    """label and categorize character images with the mouse and keyboard"""

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

    # assumes the

    patch_width = chars[0].data[0].shape[1]
    patch_height = chars[0].data[0].shape[0]

    blank_patch = np.zeros((patch_width, patch_height, 3), dtype=np.uint8)

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
        bmps = [x.data[0] if not y else blank_patch
                for x, y in chars_working]
        patch_im = util.patch_image(bmps, 16, 2)
        cv2.imshow("characters", patch_im)

    cv2.namedWindow("characters", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("characters", on_mouse, 0)
    draw()

    while True:
        if len(chars_done) == 0 and len([x for x, y in chars_working if not y]) == 0:
            break
        key = cv2.waitKey(60000)
        if key == 27:
            break
        elif key == 9:
            print("toggling type mode")
            type_mode = not type_mode
            continue
        elif key == 13:
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


def main():
    """main program"""

    label_mode = False
    label_ml = True

    patch_width = 96
    patch_height = 96

    pad = lambda x: charclassml.pad_char_bmp(x, patch_width, patch_height)

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
