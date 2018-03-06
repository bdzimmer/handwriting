"""

Unit tests for config file reading and writing.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import os
import unittest
import tempfile

import attr

from handwriting import config

VERBOSE = False


class TestsConfig(unittest.TestCase):
    """Unit tests for config file reading and writing."""

    def test_save_load(self):
        """test save and load functions"""

        @attr.s
        class TestClass:
            x = attr.ib()
            y = attr.ib()
            z = attr.ib()

        config_original = TestClass(1.0, 2, ["3", 4, 5.0])

        temp_file, temp_filename = tempfile.mkstemp(".json")
        os.close(temp_file)

        config.save(config_original, temp_filename)
        config_loaded = config.load(TestClass, temp_filename)

        if VERBOSE:
            print(temp_filename)
            print(config_original)
            print(config_loaded)

        self.assertEqual(config_original, config_loaded)
        os.remove(temp_filename)
