from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
)
from datapreparation.kitti360pose.utils import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell


class Kitti360CoarseDataset(Kitti360BaseDataset):
    def __init__(
        self,
        base_path,
        scene_name,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Dataset variant for coarse module training.
        Returns one item per pose.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_name: Scene name
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        super().__init__(base_path, scene_name)
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses

        self.sample_close_cell = sample_close_cell
        self.cell_centers = np.array([cell.get_center()[0:2] for cell in self.cells])

    def __getitem__(self, idx):
        pose = self.poses[idx]

        # TODO/NOTE: If it doesn't work, check if there is a problem with flipping later on
        if self.sample_close_cell:
            cell_size = self.cells[0].cell_size
            dists = np.linalg.norm(self.cell_centers - pose.pose_w[0:2], axis=1)
            indices = np.argwhere(dists <= cell_size / 2).flatten()
            cell = self.cells[np.random.choice(indices)]
            assert np.linalg.norm(cell.get_center()[0:2] - pose.pose_w[0:2]) < cell_size
        else:
            cell = self.cells_dict[pose.cell_id]
        hints = self.hint_descriptions[idx]

        if self.shuffle_hints:
            hints = np.random.choice(hints, size=len(hints), replace=False)

        text = " ".join(hints)

        # NOTE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)):  # Horizontal
                pose, cell, text = flip_pose_in_cell(pose, cell, text, 1)
            if np.random.choice((True, False)):  # Vertical
                pose, cell, text = flip_pose_in_cell(pose, cell, text, -1)

        object_points = batch_object_points(cell.objects, self.transform)

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell.objects]
        object_color_indices = [COLOR_NAMES.index(obj.get_color_text()) for obj in cell.objects]

        return {
            "poses": pose,
            "cells": cell,
            "objects": cell.objects,
            "object_points": object_points,
            "texts": text,
            "cell_ids": pose.cell_id,
            "scene_names": self.scene_name,
            "object_class_indices": object_class_indices,
            "object_color_indices": object_color_indices,
            "debug_hint_descriptions": hints,  # Care: Not shuffled etc!
        }

    def __len__(self):
        return len(self.poses)


class Kitti360CoarseDatasetMulti(Dataset):
    def __init__(
        self,
        base_path,
        scene_names,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
    ):
        """Multi-scene variant of Kitti360CoarseDataset.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_names: List of scene names
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.sample_close_cell = sample_close_cell
        self.datasets = [
            Kitti360CoarseDataset(
                base_path, scene_name, transform, shuffle_hints, flip_poses, sample_close_cell
            )
            for scene_name in scene_names
        ]

        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For eval

        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells)  # IDs should not repeat

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} descriptions for {num_poses} unique poses, {len(self.all_cells)} cells, flip {self.flip_poses}, close-cell {self.sample_close_cell}"

    # def get_known_words(self):
    #     known_words = []
    #     for ds in self.datasets:
    #         known_words.extend(ds.get_known_words())
    #     return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform)


class Kitti360CoarseCellOnlyDataset(Dataset):
    def __init__(self, cells: List[Cell], transform):
        """Dataset to return only the cells for encoding during evaluation
        NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
        """
        super().__init__()

        self.cells = cells
        self.transform = transform

    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        object_points = batch_object_points(cell.objects, self.transform)

        return {
            "cells": cell,
            "cell_ids": cell.id,
            "objects": cell.objects,
            "object_points": object_points,
        }

    def __len__(self):
        return len(self.cells)


if __name__ == "__main__":
    base_path = "./data/k360_30-10_scG_pd10_pc8_spY_all_nm6/"

    transform = T.FixedPoints(256)

    for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
        dataset = Kitti360CoarseDatasetMulti(
            base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
        )
        # data = dataset[0]
        # pose, cell, text = data['poses'], data['cells'], data['texts']
        # offsets = np.array([descr.offset_closest for descr in pose.descriptions])
        # hints = text.split('.')
        # pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)

        # Gather information about duplicate descriptions
        descriptors = []
        for pose in dataset.all_poses:
            mentioned = sorted(
                [f"{d.object_label}_{d.object_color_text}_{d.direction}" for d in pose.descriptions]
            )
            descriptors.append(mentioned)

        unique, counts = np.unique(descriptors, return_counts=True)
        # for d in descriptors[0:10]:
        #     print('\t',d)
        print(
            f"{len(descriptors)} poses, {len(unique)} uniques, {np.max(counts)} max duplicates, {np.mean(counts):0.2f} mean duplicates"
        )
        print("---- \n\n")