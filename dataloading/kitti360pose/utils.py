from typing import List
import numpy as np
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from copy import deepcopy

import torch
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T


OBJECT_LIST = ["building", "pole", "light", "sign", "garage", "stop", "smallpole", "lamp", "bin", "machine", "box"]
# TODO: for free orientations, possibly flip cell only, create descriptions and hints again
# OR: numeric vectors in descriptions, flip cell objects and description.direction, then create hints again
# Flip pose, too?
def flip_pose_in_cell(pose: Pose, cell: Cell, text, direction, hints=None, offsets=None):
    """Flips the cell horizontally or vertically
    CARE: Needs adjustment for non-compass directions
    CARE: Description.object_closest_point is flipped but direction in description is not flipped.

    Args:
        pose (Pose): The pose to flip, is copied before modification
        cell (Cell): The cell to flip, is copied before modification
        text (str): The text description to flip
        direction (int): Horizontally (+1) or vertically (-1)

    Returns:
        Pose: flipped pose
        Cell: flipped cell
        str: flipped text
    """
    assert direction in (-1, 1)
    assert sum([hints is None, offsets is None]) != 1  # Either both or none

    pose = deepcopy(pose)
    cell = deepcopy(cell)
    if offsets is not None:
        offsets = offsets.copy()

    if direction == 1:  # Horizontally
        pose.pose[0] = 1.0 - pose.pose[0]
        for obj in cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for descr in pose.descriptions:
            descr.closest_point[0] = 1.0 - descr.closest_point[0]

        text = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )                   

        if hints is not None:
            hints = [
                hint.replace("east", "east-flipped")
                .replace("west", "east")
                .replace("east-flipped", "west")
                for hint in hints
            ]
            offsets[:, 0] *= -1

    elif direction == -1:  # Vertically
        pose.pose[1] = 1.0 - pose.pose[1]
        for obj in cell.objects:
            obj.xyz[:, 1] = 1 - obj.xyz[:, 1]
        for descr in pose.descriptions:
            descr.closest_point[1] = 1.0 - descr.closest_point[1]

        text = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )
                    
        if hints is not None:
            hints = [
                hint.replace("north", "north-flipped")
                .replace("south", "north")
                .replace("north-flipped", "south")
                for hint in hints
            ]
            offsets[:, 1] *= -1

    assert "flipped" not in text

    if hints is not None:
        return pose, cell, text, hints, offsets
    else:
        return pose, cell, text


def batch_object_points(objects: List[Object3d], transform, use_fps=False, use_gather=False, use_knn=False, keep_num=0):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire batch of cells into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform: PyG-Transform
    """
    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    if use_knn:
        assert use_fps
        rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
        xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


        if use_gather:
            xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
        else:
            xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

        return torch.cat(xyz)

    else:
        
        if use_fps:
            rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
            xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


            if use_gather:
                xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
            else:
                xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

            data_list = [
                Data(
                    x=rgb_i.squeeze(0), pos=xyz_i.squeeze(0)
                )
                for rgb_i, xyz_i in zip(rgb, xyz)
            ]

        else:
            data_list = [
                    Data(
                        x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float)
                    )
                    for obj in objects
                ]
          

        for i in range(len(data_list)):
            data_list[i] = transform(data_list[i])

        assert len(data_list) >= 1
        batch = Batch.from_data_list(data_list)
    return batch
