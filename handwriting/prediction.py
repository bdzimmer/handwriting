# -*- coding: utf-8 -*-
"""

Data structures for samples and predictions.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import attr


@attr.s
class Sample(object):
    """struct for holding data, prediction results, and source data"""
    data = attr.ib()
    result = attr.ib()
    trust = attr.ib()
    verified = attr.ib()

    def copy(self, **changes):
        """copy self with changes"""
        return attr.assoc(self, **changes)


# TODO: get rid of Prediction
@attr.s
class Prediction(object):
    """struct for holding predictions and relevant data"""
    data = attr.ib()
    result = attr.ib()
    trust = attr.ib()
    verified = attr.ib()

    def copy(self, **changes):
        """copy self with changes"""
        return attr.assoc(self, **changes)
