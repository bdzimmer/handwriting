# -*- coding: utf-8 -*-
"""

Unit tests for character position finding module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import unittest

from handwriting import findletters

class TestsFindLetters(unittest.TestCase):

    """Unit tets for character position finding module."""

    def test_jaccard_index(self):
        """test jaccard index (intersection over union) calculation"""

        self.assertEqual(
            0.0,
            findletters.jaccard_index((0, 3), (3, 6)))
        self.assertEqual(
            0.0,
            findletters.jaccard_index((3, 6), (0, 3)))

        self.assertAlmostEqual(
            0.2,
            findletters.jaccard_index((0, 3), (2, 5)))
        self.assertAlmostEqual(
            0.2,
            findletters.jaccard_index((0, 3), (2, 5)))

        self.assertAlmostEqual(
            1.0,
            findletters.jaccard_index((0, 3), (0, 3)))

        self.assertAlmostEqual(
            0.0,
            findletters.jaccard_index((115, 145), (0, 5)))
