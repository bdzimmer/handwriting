"""

Run full recognition process.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import sys

import cv2

from handwriting import findlines, findwords, findletters, extract, util
from handwriting import analysisimage
from handwriting.prediction import Sample


VISUALIZE = True


def current_best_process():

    """construct the current best recognition process"""

    # build the individual process functions
    px_above, px_below = 72, 32

    find_line_poss = lambda im: findlines.find(im)[1]
    extract_line = lambda lpos, im: extract.extract_line_image(
        lpos, im, px_above, px_below)

    # find_word_poss = findwords.find_thresh
    find_word_poss = lambda x: findwords.find_conc_comp(x, merge=True, merge_tol=8)
    extract_word = lambda wpos, im: im[:, wpos[0]:wpos[1]]
    # find_char_poss = findletters.find_thresh_peaks # oversegments
    find_char_poss = lambda x: findwords.find_conc_comp( # undersegments
        x, merge=False)
    extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]
    # def add(x, y):
    #     return x[0] + y[0], x[0] + y[1]
    # def find_char_poss(im):
    #     """helper"""
    #     init_poss =  findwords.find_conc_comp(im, merge=False)
    #     return [add(x, y) for x in init_poss
    #             for y in findletters.find_thresh_peaks(extract_char(x, im))]

    print("loading models...", end="")
    classify_characters = util.load_dill("models/classify_characters.pkl")[0]
    classify_char = lambda x: classify_characters([x])[0]
    print("done")

    # put the pieces together
    process = build_process(
        find_line_poss, extract_line,
        find_word_poss, extract_word,
        find_char_poss, extract_char,
        classify_char)

    return process


def build_process(
        find_line_poss, extract_line,
        find_word_poss, extract_word,
        find_char_poss, extract_char,
        classify_char):

    """build full recognition process"""

    def process_image(image):
        """process an image"""
        image_sample = Sample(image, [], 0.0, False)

        for line_pos in find_line_poss(image):
            line_pos_sample = process_line_position(line_pos, image)
            image_sample.result.append(line_pos_sample)
            print(".", end="", flush=True)
        print()
        return image_sample

    def process_line_position(line_pos, image):
        """process a line position"""
        line_pos_sample = Sample(line_pos, None, 0.0, False)

        line_im = extract_line(line_pos, image)
        line_im_sample = process_line_image(line_im)
        line_pos_sample.result = line_im_sample
        return line_pos_sample

    def process_line_image(line_im):
        """process a line image"""
        line_im_sample = Sample(line_im, [], 0.0, False)

        for word_pos in find_word_poss(line_im):
            word_pos_sample = process_word_position(word_pos, line_im)
            line_im_sample.result.append(word_pos_sample)
        return line_im_sample

    def process_word_position(word_pos, line_im):
        """process a word position"""
        word_pos_sample = Sample(word_pos, None, 0.0, False)

        word_im = extract_word(word_pos, line_im)
        word_im_sample = process_word_image(word_im)
        word_pos_sample.result = word_im_sample
        return word_pos_sample

    def process_word_image(word_im):
        """process a word image"""
        word_im_sample = Sample(word_im, [], 0.0, False)

        for char_pos in find_char_poss(word_im):
            char_pos_sample = process_char_position(char_pos, word_im)
            word_im_sample.result.append(char_pos_sample)
        return word_im_sample

    def process_char_position(char_pos, word_im):
        """process a character position"""
        char_pos_sample = Sample(char_pos, None, 0.0, False)

        char_im = extract_char(char_pos, word_im)
        char_im_sample = process_char_image(char_im)
        char_pos_sample.result = char_im_sample
        return char_pos_sample

    def process_char_image(char_im):
        """process a character image"""
        char_im_sample = Sample(char_im, None, 0.0, False)

        char = classify_char(char_im)
        char_im_sample.result = char
        return char_im_sample

    return (
        process_image,
        process_line_position,
        process_word_position,
        process_char_position)


def main(argv):

    """main program"""

    if len(argv) < 2:
        print("usage: driver image")
        sys.exit()

    input_filename = argv[1]

    print("input file name:", input_filename)
    image = cv2.imread(input_filename)

    process = current_best_process()[0]

    # do the processing
    image_sample = process(image)

    if VISUALIZE:
        for line_pos in image_sample.result:
            img = analysisimage.LineAnalysisImage(line_pos.result).image
            cv2.namedWindow("line analysis", cv2.WINDOW_NORMAL)
            cv2.imshow("line analysis", img)
            key = cv2.waitKey()
            if key == 27:
                break
        cv2.destroyWindow("line analysis")

    def join_words(words):
        """helper"""
        return " ".join(["".join(x) for x in words])
    chars_remove = "`"
    line_results = [join_words([[char_pos.result.result for char_pos in word_pos.result.result]
                                for word_pos in line_pos.result.result])
                    for line_pos in image_sample.result]
    line_results = [x.translate({ord(c): None for c in chars_remove})
                    for x in line_results]

    print("\n\n")
    for line_result in line_results:
        print(line_result)
    print("\n\n")

    print("writing output file...", end="", flush=True)
    util.save(image_sample, input_filename + ".sample.pkl.0")
    print("done")


if __name__ == "__main__":
    main(sys.argv)
