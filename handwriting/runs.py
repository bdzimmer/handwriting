# -*- coding: utf-8 -*-
"""

Generate commands and configs for codalab runs.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import os
import shutil

import attr

from handwriting import run_charclassml, config as cf

DATE = "20180610"

RUN_COMMAND_PREFIX = "source activate handwriting36; export OMP_NUM_THREADS=1; python -m "
CODE_BUNDLE = "code"
DATA_BUNDLE = "data"
CONFIG_BUNDLE = "config_charclass_" + DATE
PYTHON_MODULE = "handwriting.run_charclassml"
MODULE_ARGS = "train"
CONFIG_DEFAULT_FILENAME = "config/charclass_20180523.json"

OUTPUT_DIR = "C:/Ben/code/codalab/" + DATE

CONFIG_DEFAULT = run_charclassml.load_config(CONFIG_DEFAULT_FILENAME)


def main():
    """main program"""

    def update(x, **kwargs):
        """functional dict update"""
        res = dict(x)
        for k, v in kwargs.items():
            res[k] = v
        return res

    # make a subdirectory in the output dir for the config bundle
    config_bundle_dirname = os.path.join(OUTPUT_DIR, CONFIG_BUNDLE)
    if not os.path.exists(config_bundle_dirname):
        os.makedirs(config_bundle_dirname)

    # dictionary of configs
    configs = {
        "reg_0.001": attr.assoc(
            CONFIG_DEFAULT,
            nn_opt=update(CONFIG_DEFAULT.nn_opt, weight_decay=0.001)),
        "reg_0.005": attr.assoc(
            CONFIG_DEFAULT,
            nn_opt=update(CONFIG_DEFAULT.nn_opt, weight_decay=0.005)),
        "reg_0.01": attr.assoc(
            CONFIG_DEFAULT,
            nn_opt=update(CONFIG_DEFAULT.nn_opt, weight_decay=0.01))
    }

    # save config files to bundle subdir
    for k, v in configs.items():
        config_filename = os.path.join(config_bundle_dirname, k + ".json")
        cf.save(v, config_filename)

    # TODO: zip bundle subdir
    shutil.make_archive(config_bundle_dirname, "zip", config_bundle_dirname)

    # generate a text file of run commands
    runs_filename = os.path.join(OUTPUT_DIR, "runs.txt")

    with open(runs_filename, "w") as runs_file:
        for k in configs.keys():
            config_filename = CONFIG_BUNDLE + "/" + k + ".json"
            run = " ".join([
                "run",
                "handwriting:" + CODE_BUNDLE + "/handwriting",
                ":" + DATA_BUNDLE,
                ":" + CONFIG_BUNDLE,
                "\"" + " ".join([
                    RUN_COMMAND_PREFIX,
                    PYTHON_MODULE,
                    MODULE_ARGS,
                    config_filename,
                    "model.pkl"
                ]) + "\"",
                "-n " + "run_" + k + "_" + DATE,
                "--request-docker-image bdzimmer/handwriting:0.1",
                "--request-memory 16g"
            ]) + "\n"
            print(run, file=runs_file)


if __name__ == "__main__":
    main()
