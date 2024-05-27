"""Methods for selecting objects inside a cell to describe a pose with.
"""

from typing import List

import numpy as np
import random
from datapreparation.kitti360pose.imports import Object3d

# TODO: after debug, activate shuffling in direction and class


def get_direction(obj, pose):
    closest_point = obj.get_closest_point(pose)
    obj2pose = pose - closest_point
    if np.linalg.norm(obj2pose[0:2]) < 0.05:  # Before: 0.015
        direction = "on-top"
    else:
        if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] >= 0:
            direction = "east"
        if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] <= 0:
            direction = "west"
        if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] >= 0:
            direction = "north"
        if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] <= 0:
            direction = "south"
    return direction


def get_direction_noOntop(obj, pose):
    obj2pose = pose[0:2] - obj.get_center()[0:2]
    if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] >= 0:
        direction = "east"
    if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] <= 0:
        direction = "west"
    if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] >= 0:
        direction = "north"
    if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] <= 0:
        direction = "south"
    return direction


def select_objects_closest(objects: List[Object3d], pose, num_mentioned):
    """Simply pick the closest-by objects"""
    distances = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in objects], axis=1)
    indices = np.argsort(distances)[0:num_mentioned]
    return [objects[i] for i in indices]


def select_objects_direction(objects: List[Object3d], pose, num_mentioned):
    """Try to pick objects equally distributed among the possible directions."""
    directions = [get_direction(obj, pose) for obj in objects]
    direction_indices = {d: [] for d in directions}
    for idx, d in enumerate(directions):
        direction_indices[d].append(idx)

    keys = list(direction_indices.keys())
    # random.shuffle(keys)

    offset = 0
    indices = []
    while len(indices) < num_mentioned:
        for key in keys:
            value = direction_indices[key]
            if len(value) > offset:
                indices.append(value[offset])
        offset += 1
    indices = indices[0:num_mentioned]
    return [objects[i] for i in indices]


def select_objects_class(objects: List[Object3d], pose, num_mentioned):
    """Try to describe as many classes as possible."""
    class_indices = {obj.label: [] for obj in objects}
    for idx, obj in enumerate(objects):
        class_indices[obj.label].append(idx)

    keys = list(class_indices.keys())
    # random.shuffle(keys)

    offset = 0
    indices = []
    while len(indices) < num_mentioned:
        for key in keys:
            value = class_indices[key]
            if len(value) > offset:
                indices.append(value[offset])
        offset += 1
    indices = indices[0:num_mentioned]
    return [objects[i] for i in indices]


def select_objects_random(objects: List[Object3d], pose, num_mentioned):
    """Simply pick random objects"""
    return list(np.random.choice(objects, size=num_mentioned, replace=False))
