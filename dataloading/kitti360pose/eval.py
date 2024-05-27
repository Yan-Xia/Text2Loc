from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict

import torch_geometric.transforms as T

from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from datapreparation.kitti360pose.utils import sentence_style_t, sentence_style_n, sentence_style_s, sentence_style_w, sentence_style_e
from dataloading.kitti360pose.poses import batch_object_points
from dataloading.kitti360pose.base import Kitti360BaseDataset


class Kitti360FineEvalDataset(Dataset):
    def __init__(self, poses: List[Pose], cells: List[Cell], transform, args):
        """Dataset to evaluate the fine module in isolation.
        Needed to include recall, precision and offset accuracy metrics.

        Args:
            poses (List[Pose]): List of poses
            cells (List[Cell]): List of cells
            transform: PyG transform to apply to object points
            args: Global script arguments for evaluation
        """
        super().__init__()
        self.poses = poses
        self.transform = transform
        self.args = args

        self.cells_dict = {cell.id: cell for cell in cells}

        print(
            f"Kitti360FineEvalDataset: {len(self)} poses, {len(cells)} cells, pad {args.pad_size}"
        )

    def load_pose_and_cell(self, pose: Pose, cell: Cell):
        assert pose.cell_id == cell.id
        assert len(pose.descriptions) == self.args.num_mentioned

        # Padded version here
        matched_ids = []
        for descr in pose.descriptions:
            matched_ids.append(descr.object_id if descr.is_matched else None)

        cell_objects_dict = {obj.id: obj for obj in cell.objects}

        # Gather offsets for oracle
        pose_in_cell = (pose.pose_w - cell.bbox_w[0:3])[0:2] / cell.cell_size
        oracle_offsets = []
        for descr in pose.descriptions:
            if descr.is_matched:
                # oracle_offsets.append(descr.best_offset_center[0:2])
                obj = cell_objects_dict[descr.object_id]
                oracle_offsets.append(pose_in_cell - obj.get_center()[0:2])
            else:
                oracle_offsets.append(descr.offset_center)

        # Gather the objects and matches
        objects = []
        matches = []
        for obj_idx, obj in enumerate(cell.objects):
            objects.append(obj)
            if obj.id in matched_ids:
                hint_idx = matched_ids.index(obj.id)
                matches.append((obj_idx, hint_idx))

            if len(objects) >= self.args.pad_size:
                break

        # Pad if needed
        while len(objects) < self.args.pad_size:
            objects.append(Object3d.create_padding())
        assert len(objects) == self.args.pad_size

        matches = np.array(matches)
        assert len(matches) <= len(matched_ids)  # Some matched objects can be cut-off

        return {
            "poses": pose,
            "cells": cell,
            "objects": objects,
            "object_points": batch_object_points(objects, self.transform),
            "matches": matches,
            "hint_descriptions": Kitti360BaseDataset.create_hint_description(pose, None),
            "offsets_best_center": np.array(oracle_offsets),
        }

    def __getitem__(self, idx: int):
        pose = self.poses[idx]
        cell = self.cells_dict[pose.cell_id]

        return self.load_pose_and_cell(pose, cell)

    def __len__(self):
        return len(self.poses)

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


class Kitti360TopKDataset(Dataset):
    def __init__(self, poses: List[Pose], cells: List[Cell], retrievals, transform, args,):
        """Dataset to rune the fine module on one query against multiple cells.
        Return a "batch" of each pose with each of the corresponding top-k retrieved cells.

        Args:
            poses (List[Pose]): List of poses
            cells (List[Cell]): List of cells
            retrievals: List of lists of retrievals: [[cell_id_0, cell_id_1, ...], ...]
            transform: PyG transform for object points
            args: Global evaluation arguments
        """
        super().__init__()
        self.poses = poses
        self.retrievals = retrievals
        assert len(poses) == len(retrievals)
        assert len(retrievals[0]) == max(args.top_k), "Retrievals where not trimmed to max(top_k)"
        assert len(poses) != len(cells)

        self.cells_dict = {cell.id: cell for cell in cells}
        assert len(self.cells_dict) == len(cells), "Cell-IDs are not unique"

        self.transform = transform
        self.args = args

        print(
            f"Kitti360TopKDataset: {len(self.poses)} poses, {len(cells)} cells, pad {args.pad_size}"
        )

    def load_pose_and_cell(self, pose: Pose, cell: Cell, idx: int):
        cell = deepcopy(cell)

        objects = cell.objects

        # Cut-off objects
        if len(objects) > self.args.pad_size:
            # print('Objects overflow: ', len(objects))
            objects = objects[0 : self.args.pad_size]

        # Pad objects
        while len(objects) < self.args.pad_size:
            objects.append(Object3d.create_padding())

        object_points = batch_object_points(objects, self.transform)

        hints = Kitti360BaseDataset.create_hint_description(pose, None)

        text = " ".join(hints)

        return {
            "poses": pose,
            "objects": objects,
            "object_points": object_points,
            "hint_descriptions": hints,
            "texts": text,
            "cells": cell,
        }

    def __getitem__(self, idx):
        """Return a "batch" of the pose at idx with each of the corresponding top-k retrieved cells"""
        pose = self.poses[idx]
        retrievals = self.retrievals[idx]
                

        return Kitti360TopKDataset.collate_append(
            [self.load_pose_and_cell(pose, self.cells_dict[cell_id], idx) for cell_id in retrievals]
        )

    # NOTE: returns the number of poses, each item has max(top_k) samples!
    def __len__(self):
        return len(self.poses)

    def collate_append(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

    # def collate_extend(data):
    #     batch = {}
    #     for key in data[0].keys():
    #         batch[key] = []
    #         for i in range(len(data)):
    #             assert isinstance(data[i][key], list)
    #             batch[key].extend(data[i][key])
    #     return batch


if __name__ == "__main__":
    from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti

    base_path = "./data/k360_cs30_cd15_scY_pd10_pc1_spY_closest"
    folder_name = "2013_05_28_drive_0003_sync"

    args = EasyDict(pad_size=16, top_k=(1, 3, 5))

    transform = T.FixedPoints(256)
    dataset_coarse = Kitti360CoarseDatasetMulti(
        base_path,
        [
            folder_name,
        ],
        transform,
        shuffle_hints=False,
        flip_poses=False,
    )

    retrievals = []
    for i in range(len(dataset_coarse.all_poses)):
        retrievals.append([dataset_coarse.all_cells[k].id for k in range(max(args.top_k))])

    dataset = Kitti360TopKDataset(
        dataset_coarse.all_poses, dataset_coarse.all_cells, retrievals, transform, args
    )
    data = dataset[0]

    loader = DataLoader(dataset, batch_size=2, collate_fn=Kitti360TopKDataset.collate_append)
    batch = next(iter(loader))
