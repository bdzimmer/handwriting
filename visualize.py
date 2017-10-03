"""

Visualize results without verification process.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import pickle

import annotate
import extract


def main():

    """main program"""

    input_filename = "20170929_1.png"
    preds_filename = input_filename + ".lgap.pkl"

    print("loading predictions from file")
    with open(preds_filename, "rb") as preds_file:
        preds = pickle.load(preds_file)

    preds_verified = [pred for pred in preds if pred.verified]

    letter_images = [y
                     for x in preds_verified
                     for y in extract.extract_letters(x.data, x.result)]

    for idx, letter_image in enumerate(letter_images):
        print(idx, "/", len(letter_images))
        letter = annotate.annotate_letter(letter_image, None)
        print("letter:", letter)


if __name__ == "__main__":
    main()
