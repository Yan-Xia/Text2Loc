from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
import json

import torch
from torch.utils.data import Dataset, DataLoader

from datapreparation.kitti360pose.utils import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    CLASS_TO_INDEX,
)

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)


class Kitti360BaseDataset(Dataset):
    def __init__(self, base_path, scene_name):
        """Base dataset for loading Kitti360Pose data.

        Args:
            base_path: Base path for the Kitti360Pose scenes
            scene_name: Name of the scene to load
        """
        self.scene_name = scene_name
        self.cells = pickle.load(
            open(osp.join(base_path, "cells", f"{scene_name}.pkl"), "rb")
        )  # Also use objects from here for classification
        self.cells_dict = {cell.id: cell for cell in self.cells}

        cell_ids = [cell.id for cell in self.cells]
        assert len(np.unique(cell_ids)) == len(cell_ids)

        self.poses = pickle.load(open(osp.join(base_path, "poses", f"{scene_name}.pkl"), "rb"))

        self.class_to_index = CLASS_TO_INDEX
        self.hint_descriptions = [
            Kitti360BaseDataset.create_hint_description(pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]


    def __getitem__(self, idx):
        raise Exception("Not implemented: abstract class.")

    def create_hint_description(pose: Pose, cell: Cell):
        hints = []
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')           
            hints.append(
                f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
            )
        return hints

    def get_known_classes(self):
        return list(self.class_to_index.keys())

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace(".", "").replace(",", "").lower().split())
        return list(np.unique(words))

    def __len__(self):
        raise Exception("Not implemented: abstract class.")

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


if __name__ == "__main__":
    base_path = "./data/k360_decouple"
    folder_name = "2013_05_28_drive_0003_sync"

    dataset = Kitti360BaseDataset(base_path, folder_name)
