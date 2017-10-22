"""

Run full recognition process.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import sys

import attr
import cv2
import numpy as np

from handwriting import findlines, findwords, findletters, extract, util

VISUALIZE = True


# note that this is not "frozen"
# experimenting with a mutable tree structure
@attr.s
class Sample:
    """struct for holding data, prediction results, and source data"""
    parent = attr.ib()
    data = attr.ib()
    result = attr.ib()
    trust = attr.ib()
    verified = attr.ib()
    modified = attr.ib()

    def copy(self, **changes):
        """copy self with changes"""
        return attr.assoc(self, **changes)


def line_sample_analysis_image(line_im_sample):
    """Generate a line analysis image from the appropriate Sample"""

    line_im = line_im_sample.data

    word_poss = [x.data for x in line_im_sample.result]
    word_ims = [x.result.data for x in line_im_sample.result]

    words_cposs = [y.data for x in line_im_sample.result
                   for y in x.result.result]

    words_char_ims = [[y.result.data for y in x.result.result]
                      for x in line_im_sample.result]

    words_chars = [[y.result.result for y in x.result.result]
                   for x in line_im_sample.result]

    im = line_analysis_image(
        line_im,
        word_poss, word_ims,
        words_cposs, words_char_ims,
        words_chars)
    return im



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

    # corrently wposs and words_cposs are not used in the image

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
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
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
        """helper"""
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


def char_sample_analysis_image():
    """illustrate the parent data of a character classification"""

    # everything required to generate this image should eventually be
    # available via a character sample and should be stored in the pickle file.

    # TODO: implement char sample analysis image

    return None


def current_process():

    """construct the current best recognition process"""

    # build the individual process functions
    px_above, px_below = 72, 32

    find_line_poss = lambda im: findlines.find(im)[1]
    extract_line = lambda lpos, im: extract.extract_line_image(
        lpos, im, px_above, px_below)

    # find_word_poss = findwords.find_thresh
    find_word_poss = lambda x: findwords.find_conc_comp(x, merge_tol=8)
    extract_word = lambda wpos, im: im[:, wpos[0]:wpos[1]]

    find_char_poss = findletters.find_thresh_peaks
    extract_char = lambda cpos, im: im[:, cpos[0]:cpos[1]]

    print("loading models...", end="")
    classify_characters, _ = util.load_dill("char_class_svc.pkl")
    classify_char = lambda x: classify_characters([(x,)])[0]
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
        image_sample = Sample(None, image, [], 0.0, False, False)

        for line_pos in find_line_poss(image):
            line_pos_sample = process_line_position(line_pos, image)
            image_sample.result.append(line_pos_sample)
            print(".", end="", flush=True)
        print()
        return image_sample

    def process_line_position(line_pos, image):
        """process a line position"""
        line_im = extract_line(line_pos, image)
        line_pos_sample = Sample(line_im, line_pos, None, 0.0, False, False)
        line_im_sample = process_line(line_im)
        line_pos_sample.result = line_im_sample
        return line_pos_sample

    def process_line(line_im):
        """process a line"""
        line_im_sample = Sample(None, line_im, [], 0.0, False, False)

        for word_pos in find_word_poss(line_im):
            word_im = extract_word(word_pos, line_im)

            word_pos_sample = Sample(line_im_sample, word_pos, None, 0.0, False, False)
            line_im_sample.result.append(word_pos_sample)
            word_im_sample = process_word(word_im)
            word_pos_sample.result = word_im_sample
        return line_im_sample

    # TODO: process_word_position function

    def process_word(word_im):
        """process a word"""
        word_im_sample = Sample(None, word_im, [], 0.0, False, False)

        for char_pos in find_char_poss(word_im):
            char_im = extract_char(char_pos, word_im)

            char_pos_sample = Sample(word_im_sample, char_pos, None, 0.0, False, False)
            word_im_sample.result.append(char_pos_sample)
            char_im_sample = process_char(char_im)
            char_pos_sample.result = char_im_sample

        return word_im_sample

    def process_char(char_im):
        """process a character"""
        char_im_sample = Sample(None, char_im, None, 0.0, False, False)

        char = classify_char(char_im)
        char_im_sample.result = char
        return char_im_sample

    return (
        process_image,
        process_line,
        process_line_position,
        process_word,
        process_char)


def main(argv):

    """main program"""

    if len(argv) < 2:
        print("usage: driver image")
        sys.exit()

    input_filename = argv[1]

    print("input file name:", input_filename)
    image = cv2.imread(input_filename)

    process = current_process()[0]

    # do the processing
    image_sample = process(image)

    if VISUALIZE:
        for line_pos in image_sample.result:
            im = line_sample_analysis_image(line_pos.result)
            cv2.namedWindow("line analysis", cv2.WINDOW_NORMAL)
            cv2.imshow("line analysis", im)
            cv2.waitKey()
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

    print("writing output file...", end="")
    util.save(image_sample, input_filename + ".sample.pkl.0")
    print("done")

if __name__ == "__main__":
    main(sys.argv)
