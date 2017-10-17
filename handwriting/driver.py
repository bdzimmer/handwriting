"""

Run full recognition process.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import sys

import attr
import cv2
import numpy as np

from handwriting import findlines, findwords, findletters
from handwriting import extract, util


VISUALIZE = True

@attr.s
class Sample(object):
    """struct for holding data, prediction results, and source data"""
    parent = attr.ib
    data = attr.ib()
    result = attr.ib()
    trust = attr.ib()
    verified = attr.ib()

    def copy(self, **changes):
        """copy self with changes"""
        return attr.assoc(self, **changes)



def line_analysis_image(
        line_im,
        wposs, word_ims,
        words_cposs, words_char_ims,
        words_chars):
    """illustrate how an entire line was analyzed"""

    # line_im  - line image

    # wposs    - list of word position tuples
    # word_ims - list of word images

    # words_cposs    - for each word, a list of char position tuples
    # words_char_ims - for each word, a list of char images

    # words_chars    - for each word, a list of classified character

    line_height = line_im.shape[0]

    hgap_large_im = np.zeros((line_height, 32, 3), dtype=np.uint8)
    hgap_small_im = np.zeros((line_height, 4, 3), dtype=np.uint8)

    def combine_ims(ims, sep_im):
        """combine images horizontally with a separating image in between"""
        if len(ims) == 0:
            return np.zeros((line_height, 0, 3), dtype=np.uint8)
        else:
            return np.hstack([y for x in ims for y in [x, sep_im]][:-1])

    all_words_im = combine_ims(word_ims, hgap_large_im)

    all_char_ims_im = combine_ims(
        [combine_ims(x, hgap_small_im)
         for x in words_char_ims], hgap_large_im)

    def char_image(ch):
        """draw character image"""
        char_im = np.zeros((line_height, 20, 3), dtype=np.uint8)
        cv2.putText(
            char_im, ch, (1, int(line_height / 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        return char_im

    all_chars_im = combine_ims(
        [combine_ims([char_image(y) for y in x], hgap_small_im)
         for x in words_chars], hgap_large_im)

    max_width = np.max([
        line_im.shape[1],
        all_words_im.shape[1],
        all_char_ims_im.shape[1],
        all_chars_im.shape[1]])

    def pad(im):
        padded_im = np.zeros((line_height, max_width, 3), dtype=np.uint8)
        start_x = int(0.5 * (max_width - im.shape[1]))
        padded_im[:, start_x:(start_x + im.shape[1]), :] = im
        return padded_im

    vgap_im = np.zeros((16, max_width, 3), dtype=np.uint8)

    res = np.vstack((
        pad(line_im),
        vgap_im,
        pad(all_words_im),
        vgap_im,
        pad(all_char_ims_im),
        vgap_im,
        pad(all_chars_im)))

    return res


def char_analysis_image():
    """illustrate the parent data of a character classification"""

    # everything required to generate this image should eventually be
    # available via a character sample and should be stored in the pickle file.

    pass


def main(argv):

    """main program"""

    if len(argv) < 2:
        print("usage: driver image")
        sys.exit()

    input_filename = argv[1]

    print("input file name:", input_filename)
    image = cv2.imread(input_filename)

    # process functions
    px_above, px_below = 72, 32

    find_lines = lambda im: findlines.find(im)[1]
    extract_line = lambda lpos, im: extract.extract_line_image(
        lpos, im, px_above, px_below)

    # find_word_poss = findwords.find_thresh
    find_word_poss = lambda x: findwords.find_conc_comp(x, merge_tol=8)
    extract_word = lambda wpos, im: im[:, wpos[0]:wpos[1]]

    find_char_poss = findletters.find_thresh_peaks
    extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]

    print("loading models...", end="")
    classify_characters, _ = util.load_dill("char_class_svc.pkl")
    print("done")

    # run the process on each line separately

    print("finding and extracting lines...")
    line_ims = [extract_line(line_pos, image)
                for line_pos in find_lines(image)]
    print("done")

    def process_line(line_im):
        """process a line"""

        word_poss = find_word_poss(line_im)
        word_ims = [extract_word(word_pos, line_im)
                    for word_pos in word_poss]

        if len(word_ims) == 0:
            return ""

        word_ims_height = [x.shape[0] for x in word_ims]
        word_ims_width = [x.shape[1] for x in word_ims]

        if False:
            print(
                "width:",
                min(word_ims_width), max(word_ims_width))
            print(
                "height:",
                min(word_ims_height), max(word_ims_height))

        words_cposs = [find_char_poss(word_im) for word_im in word_ims]

        words_char_ims = [[extract_char(cpos, y) for cpos in x]
                          for x, y in zip(words_cposs, word_ims)]

        if False:
            for char_ims in words_char_ims:
                for char_im in char_ims:
                    print(char_im.shape[1], ",", end="")
                print("_", end="")
            print()

        chars = [classify_characters([(x,) for x in char_ims])
                 for char_ims in words_char_ims
                 if len(char_ims) > 0]

        result = " ".join(["".join(x) for x in chars])
        print(".", end="", flush=True)

        if VISUALIZE:
            im = line_analysis_image(
                line_im,
                word_poss, word_ims,
                words_cposs, words_char_ims,
                chars)
            cv2.namedWindow("line analysis", cv2.WINDOW_NORMAL)
            cv2.imshow("line analysis", im)
            cv2.waitKey()

        return result

    chars_remove = "`"

    line_results = [
        process_line(line_im).translate({ord(c): None for c in chars_remove})
        for line_im in line_ims
        if line_im.shape[0] > 0]

    print("\n\n")
    for line_result in line_results:
        print(line_result)


if __name__ == "__main__":
    main(sys.argv)
