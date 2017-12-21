"""

Utilities for functional programming.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import inspect
from collections import OrderedDict


def pipe(*funcs):
    """compose single argument functions left to right (evaluated
    right to left)"""

    def piped(*args, **kwargs):
        """helper"""
        arg = funcs[0](*args, **kwargs)
        for func in funcs[1:]:
            arg = func(arg)
        return arg

    piped.__signature__ = inspect.signature(funcs[0])

    return piped


def unzip(iterable):
    """unzip a list of tuples into several lists"""
    return zip(*iterable)


def expand_params(param_values):
    """expand combinations of parameter values"""

    # param_values - list of tuples of parameter names and lists of values

    n_params = len(param_values)

    if n_params == 0:
        return []

    param_counts = [len(x[1]) for x in param_values]
    idx = [0 for _ in param_counts]

    param_sets = []
    done = False

    while not done:
        param_set = OrderedDict(
            (param_values[p_idx][0],
             param_values[p_idx][1][idx[p_idx]])
            for p_idx in range(n_params))
        param_sets.append(param_set)

        for p_idx in range(n_params):
            idx[p_idx] += 1
            if idx[p_idx] > (param_counts[p_idx] - 1):
                if p_idx == n_params - 1:
                    done = True
                idx[p_idx] = 0
            else:
                break

    return param_sets


def grid_search(
        func,
        gs_param_order=None,
        gs_callback=None,
        **kwargs):

    """call a function repeatedly using all the combinations of args specified
    by kwargs"""

    # func           - function to call with
    # gs_param_order - optional custom ordering of parameters; if None, order in
    #                  func signature
    # gs_callback    - optional function of signature param_set, result to call
    #                  after every invocation
    # **kwargs       - names and ranges of values to use

    if gs_param_order is None:
        # get names of args in function signature
        gs_param_order = function_param_order(func)

    param_values_list = []
    for param_name in gs_param_order:
        if param_name in kwargs.keys():
            param_values_list.append((param_name, kwargs[param_name]))

    param_sets = expand_params(param_values_list)

    all_results = []
    for param_set in param_sets:
        res = func(**param_set)
        all_results.append((param_set, res))
        if gs_callback is not None:
            gs_callback(param_set, res)

    return all_results


def function_param_order(func):
    """get names of args infunction signature"""
    return inspect.signature(func).parameters.keys()
