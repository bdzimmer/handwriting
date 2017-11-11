"""

Unit tests for functional programming utilities.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import unittest

from handwriting.func import pipe, unzip, expand_params, grid_search


class TestsFunc(unittest.TestCase):

    """Unit tests for functional programming utilities."""

    def test_pipe(self):
        """test pipe"""

        piped = pipe(
            lambda x: "a" + x,
            lambda x: x + "c",
            lambda x: x + "d")

        self.assertEqual(piped("b"), "abcd")


    def test_unzip(self):
        """test unzip"""

        res = list(unzip([(1, 2), (3, 4), (5, 6)]))

        # TODO: I don't like how res[0] and res[1] are tuples
        # lists would be better

        self.assertEqual(res[0], (1, 3, 5))
        self.assertEqual(res[1], (2, 4, 6))


    def test_expand_params(self):
        """test parameter expanding"""

        param_ranges = [("x", [1, 2, 3]), ("y", [4, 5, 6])]
        param_sets = expand_params(param_ranges)

        self.assertEqual(len(param_sets), 9)
        for param_set in param_sets:
            self.assertEqual(len(param_set.items()), 2)


    def test_grid_search(self):
        """test grid search"""

        # param order should be x, y from the arg names of the lambda
        res = grid_search(
            lambda x, y: x + y, x=[1, 2, 3], y=[4, 5, 6])
        self.assertEqual(len(res), 9)
        self.assertEqual(list(res[0][0].keys()), ["x", "y"])

        # param order should be y, x from the arg names of the lambda
        res = grid_search(
            lambda y, x: x + y, x=[1, 2, 3], y=[4, 5, 6])
        self.assertEqual(list(res[0][0].keys()), ["y", "x"])

        # param order should be y, x from specified param_order
        res = grid_search(
            lambda x, y: x + y, gs_param_order=["y", "x"],
            x=[1, 2, 3], y=[4, 5, 6])
        self.assertEqual(list(res[0][0].keys()), ["y", "x"])

        # test callback
        call_count = [0]
        def callback(param_set, res):
            """helper"""
            # print(param_set, ":", res)
            call_count[0] = call_count[0] + 1
        res = grid_search(
            lambda x, y: x + y, gs_callback=callback,
            x=[1, 2, 3], y=[4, 5, 6])
        self.assertEqual(call_count[0], 9)
