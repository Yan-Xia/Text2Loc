from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict
from copy import deepcopy
import pickle

import torch
from torch.utils.data import DataLoader
import time

from scipy.spatial.distance import cdist

from models.cell_retrieval import CellRetrievalNetwork

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

from datapreparation.kitti360pose.utils import SCENE_NAMES_TEST, SCENE_NAMES_VAL, KNOWN_CLASS
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360

from training.coarse import eval_epoch as eval_epoch_retrieval
from training.utils import plot_retrievals
from models.cross_matcher import get_pos_in_cell

import torch_geometric.transforms as T
import tqdm

"""
TODO:
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
"""


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}


    retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
        model, dataloader, args
    )
    retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))

    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
        )
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
            )
    model_coarse.load_state_dict(model_coarse_dic, strict = False)
    model_coarse.to(device)

    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

