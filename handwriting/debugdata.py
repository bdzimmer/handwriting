"""

Tools to help find problems in annotated data.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import cv2

from handwriting import data, util

CHARS_REMOVE = "`"


def read_text(filename):
    """get the text in a file of predictions / annotations"""

    image_sample = util.load(filename)

    def join_words(words):
        """helper"""
        return " ".join(["".join(x) for x in words])

    line_results = [join_words(
                        [[char_pos.result.result
                          for char_pos in word_pos.result.result]
                         for word_pos in line_pos.result.result])
                    for line_pos in image_sample.result]
    line_results = [x.translate({ord(c): None for c in CHARS_REMOVE})
                    for x in line_results]

    return "\n".join(line_results)


def vis_char(filename, char_to_vis):
    """find and visualize images of a specific character
    and the words that contain them"""

    image_sample = util.load(filename)

    for line_idx, line_pos in enumerate(image_sample.result):
        for word_pos in line_pos.result.result:
            for char_pos in word_pos.result.result:
                if char_pos.result.result == char_to_vis:
                    print(filename, line_idx + 1)
                    cv2.imshow("char", char_pos.result.data)
                    cv2.imshow("word", word_pos.result.data)
                    cv2.waitKey(0)


def main():
    """main program"""

    filenames = data.pages(range(5, 15))

    # for filename in filenames:
    #     vis_char(filename, "A")

    for filename in filenames:
        text = read_text(filename)
        text_filename = filename + ".txt"
        print(text_filename)
        with open(text_filename, "w") as text_file:
            text_file.write(text)


if __name__ == "__main__":
    main()
