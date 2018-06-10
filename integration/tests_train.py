# -*- coding: utf-8 -*-
"""

Integration tests for machine learning training processes.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import unittest

import os
import shutil

import attr

from handwriting import run_charclassml, run_charposml
from handwriting import config as cf


class TestML(unittest.TestCase):
    """Integration tests for machine learning training processes"""

    def test_run_charclassml(self):
        """test run_charclassml"""
        config = attr.assoc(run_charclassml.CONFIG_DEFAULT)
        config.nn_opt["max_epochs"] = 1
        config.train.idxs = [1]
        config.train.do_balance = True
        config.train.balance_size = 64
        config.dev.idxs = [2]
        config.test.idxs = [3]
        self._test_training_module(
            run_charclassml,
            "charclass",
            config)

    def test_run_charposml(self):
        """test run_charposml"""
        run_charposml.VERBOSE = True
        config = attr.assoc(run_charposml.CONFIG_DEFAULT)
        config.train.idxs = [1]
        config.nn_opt["max_epochs"] = 1
        config.train.do_balance = True
        config.train.balance_size = 64
        config.dev.idxs = [2]
        config.test.idxs = [3]
        self._test_training_module(
            run_charposml,
            "charpos",
            config)

    def _test_training_module(self, module, sub_dirname, config):
        """helper method: run one of the training modules with config,
        testing that it produces output files"""

        # this function might not make sense later if the training modules
        # get different configurations or have different types of outputs

        work_dirname = os.path.join("integration", sub_dirname)

        if os.path.exists(work_dirname):
            shutil.rmtree(work_dirname)
        os.makedirs(work_dirname)

        # generate config file
        config_filename = os.path.join(work_dirname, "config.json")
        cf.save(config, config_filename)
        model_filename = os.path.join(work_dirname, "model.pkl")
        args = ["", "train", config_filename, model_filename]

        module.main(args)

        # assert that output model and log exist
        self.assertTrue(os.path.exists(model_filename))
        self.assertTrue(os.path.exists(model_filename + ".log.txt"))
