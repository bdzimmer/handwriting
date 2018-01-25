"""

Interactive annotations on images.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import os
import math
import pickle
import sys

import numpy as np
import cv2

from handwriting import geom


def annotate_lines(image, lines):

    """Interactively annotate lines (usually to indicate lines of text)."""

    scale_factor = 0.2

    lines = [x for x in lines]

    def draw():
        """helper"""
        disp_im = cv2.resize(image, (int(image.shape[1] * scale_factor), (int(image.shape[0] * scale_factor))))
        for lidx, line in enumerate(lines):
            lf = geom.build_line_func(*line)
            pt0 = lf(-10)
            pt1 = lf(10)
            pt_avg_scaled = (int((pt0[0] + pt1[0]) / 2.0 * scale_factor),
                             int((pt0[1] + pt1[1]) / 2.0 * scale_factor))
            cv2.line(
                disp_im,
                (int(pt0[0] * scale_factor), int(pt0[1] * scale_factor)),
                (int(pt1[0] * scale_factor), int(pt1[1] * scale_factor)),
                (255, 0, 0), 1)
            cv2.putText(
                disp_im, str(lidx), pt_avg_scaled,
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.imshow("image", disp_im)

    line = []

    def argmin_dist_to_line(x, y):
        return np.argmin([geom.point_to_line(x, y, *line) for line in lines])

    def on_mouse(event, x, y, flags, params):
        """helper"""
        x = int(x / scale_factor)
        y = int(y / scale_factor)

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(line) == 0:
                print(x, y, "->", end=" ")
                coords = [x, y]
                line.extend(coords)
            else:
                print(x, y)
                coords = [x, y]
                line.extend(coords)
                lines.append([x for x in line])
                line.clear()
                draw()
                cv2.waitKey(1)
        if event == cv2.EVENT_RBUTTONDOWN:
            # find closest line
            line_idx = argmin_dist_to_line(x, y)
            lines.pop(line_idx)
            draw()
            cv2.waitKey(1)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", on_mouse, 0)
    draw()

    while True:
        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == 8:
            lines.pop()
            draw()

    lines_sorted = sorted(lines, key=lambda x: 0.5 * (x[1] + x[3]))
    cv2.destroyWindow("image")

    return lines_sorted


def annotate_word_positions(line_image, word_positions):

    """Interactively annotate word positions."""

    scale_factor = 1.0

    wpos = [x for x in word_positions]

    def draw():
        """helper"""
        disp_image = 255 - np.copy(line_image)
        print("count:", len(wpos))
        for widx, wp in enumerate(wpos):
            for idx in np.arange(wp[0], wp[1]):
                disp_image[:, idx, 2] = 255
                cv2.putText(
                    disp_image, str(widx), (wp[0], 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            disp_image[:, wp[0], 1] = 255
            disp_image[:, wp[1] - 1, 1] = 255
        cv2.imshow("positions", cv2.resize(
            disp_image,
            (int(disp_image.shape[1] * scale_factor), int(disp_image.shape[0] * scale_factor))))

    word = []

    def on_mouse(event, x, y, flags, params):
        """helper"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(word) == 0:
                print(x, "->", end=" ", flush=True)
                word.append(int(x / scale_factor))
            else:
                print(x)
                word.append(int(x / scale_factor))
                wpos.append((word[0], word[1]))
                wpos.sort(key=lambda wp: wp[0] + wp[1])
                word.clear()
                draw()
                cv2.waitKey(1)
        if event == cv2.EVENT_RBUTTONDOWN:
            # find closest word
            word_idx = np.argmin([
                math.fabs(x / scale_factor - (wp[0] + wp[1]) / 2.0)
                for wp in wpos])
            # delete it
            wpos.pop(word_idx)
            draw()
            cv2.waitKey(1)

    cv2.namedWindow("positions", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("positions", on_mouse, 0)
    draw()

    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyWindow("positions")

    return wpos


def annotate_letter_gaps(word_image, letter_gaps):

    """Interactively annotate gaps between letters."""

    if word_image.shape[0] == 0 or word_image.shape[1] == 0:
        return None

    scale_factor = 1.0

    lgaps = [x for x in letter_gaps]

    # import util
    # util.VISUALIZE = True
    # import findletters
    # findletters.find_gaps_thresh_peaks(word_image)

    def draw():
        """helper"""
        disp_image = 255 - np.copy(word_image)
        print("gap count:", len(lgaps))
        for lidx, lgap in enumerate(lgaps):
            disp_image[:, lgap, 0:2] = 255
            cv2.putText(
                disp_image, str(lidx), (lgap, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        print()
        cv2.imshow("letter gaps", cv2.resize(
            disp_image,
            (int(disp_image.shape[1] * scale_factor), int(disp_image.shape[0] * scale_factor))))

    mouse_x = [0]

    def on_mouse(event, x, y, flags, params):
        """helper"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x)
            lgaps.append(int(x / scale_factor))
            lgaps.sort()
            draw()
            cv2.waitKey(1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # find closest word
            lgap_idx = np.argmin([
                math.fabs(x / scale_factor - lgap) for lgap in lgaps])
            # delete it
            lgaps.pop(lgap_idx)
            draw()
            cv2.waitKey(1)
        elif event == cv2.EVENT_MOUSEMOVE:
            mouse_x[0] = x

    cv2.namedWindow("letter gaps", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("letter gaps", on_mouse, 0)
    draw()

    while True:
        key = cv2.waitKey(1)

        # z and c tweak the nearest gap back and forth
        if key == ord("z") or key == ord("c"):
            lgap_idx = np.argmin([
                math.fabs(mouse_x[0] / scale_factor - lgap)
                for lgap in lgaps])
            if key == ord("z"):
                lgaps[lgap_idx] -= 1
            else:
                lgaps[lgap_idx] += 1
            draw()

        if key == 27 or key == ord("x"):
            break

    cv2.destroyWindow("letter gaps")

    return lgaps


def annotate_character(char_image, word_image, character):

    """Interactively label an image of a character."""

    scale_factor = 1.0

    def draw():
        """helper"""
        disp_image = np.copy(char_image)
        cv2.putText(
            disp_image, character, (0, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("character image", cv2.resize(
            disp_image,
            (int(disp_image.shape[1] * scale_factor), int(disp_image.shape[0] * scale_factor))))

    cv2.imshow("word image", cv2.resize(word_image, (word_image.shape[1] * 2, word_image.shape[0] * 2)))
    cv2.resizeWindow("word image", word_image.shape[1] * 2, word_image.shape[0] * 2)
    cv2.namedWindow("character image", cv2.WINDOW_NORMAL)
    draw()

    key = cv2.waitKey(600000)
    if key == 27:
        new_char = char_image
    elif key == 8: # backspace
        new_char = None
    else:
        new_char = chr(key & 0xFF)

    cv2.destroyWindow("character image")
    cv2.destroyWindow("word image")

    return new_char


def main(argv):

    """main program"""

    input_file = argv[1]

    lines_filename = input_file + ".lines.text"

    image = cv2.imread(input_file)
    # im2 = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

    if os.path.exists(lines_filename):
        with open(lines_filename, "rb") as lines_file:
            lines = pickle.load(lines_file)
    else:
        lines = []

    lines = annotate_lines(image, lines)

    with open(lines_filename, "wb") as lines_file:
        pickle.dump(lines, lines_file)


if __name__ == "__main__":
    main(sys.argv)
