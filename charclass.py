# -*- coding: utf-8 -*-
"""

Experimentation with more complex methods of interactive character annotation.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import cv2
import numpy as np

import util


def _pad_char_bmp(char_bmp, width, height):
    """pad char bitmap in a larger bitmap"""

    char_bmp = char_bmp[32:, :]

    new_bmp = np.ones((height, width, 3), dtype=np.uint8) * 255

    xoff = int((width - char_bmp.shape[1]) / 2)
    yoff = int((height - char_bmp.shape[0]) / 2)

    new_bmp[yoff:(yoff + char_bmp.shape[0]), xoff:(xoff + char_bmp.shape[1])] = char_bmp

    return new_bmp


def main():
    """main program"""

    patch_width = 96
    patch_height = 96

    input_filename = "20170929_1.png.character.pkl"
    with open(input_filename, "rb") as input_file:
        chars = pickle.load(input_file)

    if True:

        unique_chars = sorted(list(set([x.result for x in chars if x.result is not None])))
        for cur_char in unique_chars:
            print(cur_char)
            cur_preds = [x for x in chars if x.result == cur_char]
            bmps = [_pad_char_bmp(x.data[0], patch_width, patch_height)
                    for x in cur_preds]
            patch_im = util.patch_image(bmps)
            cv2.namedWindow("characters", cv2.WINDOW_NORMAL)
            cv2.imshow("characters", patch_im)
            cv2.waitKey()

    else:

        chars_verified = [x for x in chars if x.verified]
        chars_working = [x for x in chars if not x.verified]
        new_char = ["Z"]
        idx = [0]
        type_mode = False

        blank_patch = np.zeros((patch_height, patch_width, 3), dtype=np.uint8)

        def on_mouse(event, x, y, flags, params):
            """helper"""
            if event == cv2.EVENT_LBUTTONDOWN:
                idx[0] = int(y / patch_height) * 16 + int(x / patch_width)
                print(idx)
                pred = chars_working[idx[0]]
                pred_new = pred.copy(result=new_char[0], verified=True)
                chars_working[idx[0]] = pred_new
                draw()

        def draw():
            """helper"""
            print("total unverified characters:", len([x for x in chars_working if not x.verified]))
            bmps = [_pad_char_bmp(x.data[0], patch_width, patch_height) if not x.verified else blank_patch
                    for x in chars_working]
            patch_im = util.patch_image(bmps)
            cv2.imshow("characters", patch_im)

        cv2.namedWindow("characters", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("characters", on_mouse, 0)
        draw()

        while True:
            key = cv2.waitKey(60000)
            if key == 27:
                break
            elif key == 9:
                print("toggling type mode")
                type_mode = not type_mode
                continue
            elif key == 13:
                chars_verified = chars_verified + [x for x in chars_working if x.verified]
                chars_working = [x for x in chars_working if not x.verified]
                idx[0] = 0
                draw()
                continue
            elif key == 8 and type_mode: # backspace
                idx[0] = idx[0] - 1
                if idx[0] < 0:
                    idx[0] = 0
                chars_working[idx[0]] = chars_working[idx[0]].copy(verified=False)
                draw()
                continue
            elif key == 32: # space
                new_char[0] = None
                print("marking invalid")
            else:
                new_char[0] = chr(key & 0xFF)
                print("marking " + new_char[0])

            if type_mode:
                pred = chars_working[idx[0]]
                pred_new = pred.copy(result=new_char[0], verified=True)
                chars_working[idx[0]] = pred_new
                idx[0] = idx[0] + 1
                draw()

        chars_verified = chars_verified + [x for x in chars_working if x.verified]
        chars_working = [x for x in chars_working if not x.verified]

        res = chars_verified + chars_working

        with open(input_filename, "wb") as output_file:
            pickle.dump(res, output_file)


if __name__ == "__main__":
    main()
