"""

Read and write attrs objects to json files.

"""

# Copyright (c) 2018 Ben Zimmer. All rights reserved.

import json

import attr


def save(obj, output_filename):
    """save an attrs object to a json file"""
    with open(output_filename, "w") as output_file:
        json.dump(attr.asdict(obj), output_file, indent=2, sort_keys=True)


def load(cls, input_filename):
    """load an attrs object from a json file"""
    with open(input_filename, "r") as input_file:
        obj_dict = json.load(input_file)
    return cls(**obj_dict)


def pretty_print(obj):
    """pretty print an attrs object"""
    obj_dict = attr.asdict(obj)
    fields = list(attr.fields(obj.__class__))
    max_name_length = max([len(x.name) for x in fields])

    for field in fields:
        print(
            field.name + ":" + (max_name_length - len(field.name)) * " ",
            obj_dict[field.name])
