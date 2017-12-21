# -*- coding: utf-8 -*-
"""

Unit tests for machine learning utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import unittest

from handwriting import ml


class TestsML(unittest.TestCase):

    """Unit tests for machine learning utility module."""

    def test_balance(self):
        """test dataset balancing functionality"""

        data, labels = zip(*([(0.0, "a")] * 4 + [(1.0, "b")] * 1))

        balanced_data, balanced_labels = ml.balance(
            data, labels, 0.5, lambda x: x)
        balanced_grouped = dict(ml.group_by_label(
            balanced_data, balanced_labels))
        for label, group in balanced_grouped.items():
            self.assertEqual(len(group), 2)

        balanced_data, balanced_labels = ml.balance(
            data, labels, 8, lambda x: x)
        balanced_grouped = dict(ml.group_by_label(
            balanced_data, balanced_labels))
        for label, group in balanced_grouped.items():
            self.assertEqual(len(group), 8)
