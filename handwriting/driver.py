"""

Run full entire recognition process.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import sys

import attr
import cv2

from handwriting import findlines, findwords, findletters
from handwriting import extract, util


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

    find_word_poss = findwords.find_thresh
    extract_word = lambda wpos, im: im[:, wpos[0]:wpos[1]]

    find_char_poss = findletters.find_gaps_thresh_peaks
    extract_characters = lambda lgaps, im: extract.extract_letters(
        im, lgaps)

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

        word_poss = [(wpos, line_im)
                     for wpos in find_word_poss(line_im)]
        word_ims = [extract_word(word_pos, line_im)
                    for word_pos, line_im in word_poss]

        if False:
            for idx, word_im in enumerate(word_ims):
                print(idx, "/", len(word_ims))
                cv2.imshow("word", word_im)
                cv2.waitKey()

        if False:
            word_ims = [x for x in word_ims
                        if x.shape[1] > 0 and x.shape[1] <= 96]

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

        word_char_ims = [[(char_im,) for char_im in extract_characters(
            find_char_poss(word_im), word_im)]
            for word_im in word_ims]

        if False:
            for char_ims in word_char_ims:
                for char_im in char_ims:
                    print(char_im[0].shape[1], ",", end="")
                print("_", end="")
            print()

        words = [classify_characters(char_ims)
                 for char_ims in word_char_ims
                 if len(char_ims) > 0]

        result = " ".join(["".join(x) for x in words])
        print(".", end="", flush=True)

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
