# -*- coding: utf-8 -*-
"""

Data structures for predictions.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

import attr

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
