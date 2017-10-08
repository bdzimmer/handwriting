"""

Geometry utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.


import math

import numpy as np


def build_line_func(x1, y1, x2, y2):
    """build a function that parameterizes a line"""
    def line_func(t):
        """helper"""
        return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
    return line_func


def points_to_polar(x1, y1, x2, y2):

    """convert the two points on the line into something like the standard
    polar r and theta Hough transform output"""

    dy = y2 - y1
    dx = x2 - x1

    r = math.fabs(x2 * y1 - y2 * x1) / math.sqrt(dy * dy + dx * dx)
    theta = math.atan2(dy, dx)

    return r, theta


def line_line_intersection(line1, line2, is_line1, is_line2):
    """find the point of intersection between two lines or line segments"""

    # http://www-cs.ccny.cuny.edu/~wolberg/capstone/intersection/Intersection%20point%20of%20two%20lines.html

    EPS = 1e-9

    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]
    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    d = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if math.fabs(d) < EPS:
        # segments are parallel
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / d
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / d

    if (is_line1 or (ua > 0.0 and ua < 1.0)) and (is_line2 or (ub > 0.0 and ub < 1.0)):
        # intersection is in segments, not lines
        x = x1 + ua * (x2 - x1)
        y = y1 + ua * (y2 - y1)
        return x, y
    else:
        return None


def polar_to_points(r, theta):
    """convert polar form of a line to two point representation"""
    x1 = r * math.sin(theta)
    y1 = r * math.cos(theta)
    x2 = x1 + math.cos(theta)
    y2 = y1 - math.sin(theta)
    return x1, y1, x2, y2


def point_to_line(x0, y0, x1, y1, x2, y2):
    """distance from point x0, y0 to line x1, y1 to x2, y2"""
    dy = y2 - y1
    dx = x2 - x1
    return math.fabs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / math.sqrt(dy * dy + dx * dx)


def line_segment_within_image(line, shape):
    """given a line or segment, find a new line that lies on top of it where
    the points are on the edges of the image"""
    possible_intersections = [
        line_line_intersection(line, (0, shape[0] - 1, 0, 0), True, False), # left side
        line_line_intersection(line, (0, 0, shape[1] - 1, 0), True, False), # top
        line_line_intersection(line, (shape[1] - 1, 0, shape[1] - 1, shape[0] - 1), True, False), # right side
        line_line_intersection(line, (shape[1] - 1, shape[0] - 1, 0, shape[0] - 1), True, False)] # bottom
    pts = [x for x in possible_intersections if x is not None]
    return (pts[0][0], pts[0][1], pts[1][0], pts[1][1])


def line_distance(line1, line2, image_shape):
    """calculate a distance score between two lines"""

    # find the intersection of line 1 with the edges of the image
    line1_image = line_segment_within_image(line1, image_shape)

    n_points = 10

    sample_points = np.arange(0, n_points) / (n_points - 1.0)

    lf = build_line_func(*line1_image)

    line_points = [lf(x) for x in sample_points]
    distances = np.array([point_to_line(*x, *line2) for x in line_points])

    return np.sqrt(np.mean(np.square(distances))), distances, line_points


def line_list_distance(lines1, lines2, image_shape):
    """calculate a distance score between two lists of lines"""

    scores = np.zeros(len(lines1))
    closest_idxs = np.zeros(len(lines1), dtype=np.int)
    for idx, line1 in enumerate(lines1):
        # find the closest score in the other set
        line1_to_lines2 = [line_distance(line1, x, image_shape)[0] for x in lines2]
        closest_idx = np.argmin(line1_to_lines2)
        closest_idxs[idx] = closest_idx
        scores[idx] = line1_to_lines2[closest_idx]
    return np.sqrt(np.mean(np.square(scores))), scores, closest_idxs
