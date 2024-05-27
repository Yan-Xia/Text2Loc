"""Module for training the fine matching module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections

import torch_geometric.transforms as T

import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp
import tqdm

from models.cross_matcher import CrossMatch

from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti


# from datapreparation.semantic3d.imports import COLORS as COLORS_S3D, COLOR_NAMES as COLOR_NAMES_S3D
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, calc_recall_precision, calc_pose_error2

"Training Process for fine localization"
def train_epoch(model, dataloader, args):
    model.train()

    offset_lambda = args.offset_lambda
    
    stats = EasyDict(
        loss=[],
        loss_offsets=[],
        # recall=[],
        # precision=[],
        # pose_mid=[],
        # pose_mean=[],
        pose_offsets=[],
    )
        
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:

        optimizer.zero_grad()
        texts = batch["texts"]

        output = model(batch["objects"], texts, batch["object_points"])

        # loss_matching = criterion_matching(output.P, batch["all_matches"])

        # import pdb; pdb.set_trace()
        loss_offsets = criterion_offsets(
            output, torch.tensor(batch["offsets"], dtype=torch.float, device=device)
        )
        loss =  offset_lambda * loss_offsets

        try:
            loss.backward()
            optimizer.step()
        except Exception as e:
            print()
            print(str(e))
            print()
            print(batch["all_matches"])

        # recall, precision = calc_recall_precision(
        #     batch["matches"],
        #     output.matches0.cpu().detach().numpy(),
        #     output.matches1.cpu().detach().numpy(),
        # )

        stats.loss.append(loss.item())
        error = calc_pose_error2(
                batch["objects"],
                # output.matches0.detach().cpu().numpy(),
                batch["poses"],
                offsets=output.detach().cpu().numpy(),
            )
        stats.pose_offsets.append(
            error
        )
        pbar.set_postfix(loss = loss_offsets.item(), error = error)


    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


@torch.no_grad()
def eval_epoch(model, dataloader, args,):
    model.eval() 
    
    stats = EasyDict(
        # recall=[],
        # precision=[],
        # pose_mid=[],
        # pose_mean=[],
        pose_offsets=[],
    )
    
    for i_batch, batch in tqdm.tqdm(enumerate(dataloader), total = len(dataloader)):
        
        texts = batch["texts"]
        output = model(batch["objects"], texts, batch["object_points"])
        stats.pose_offsets.append(
            calc_pose_error2(
                batch["objects"],
                # output.matches0.detach().cpu().numpy(),
                batch["poses"],
                offsets=output.detach().cpu().numpy(),
            )
        )

    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")
    if not osp.isdir(f"./checkpoints/{dataset_name}/{folder_name}"):
        os.mkdir(f"./checkpoints/{dataset_name}/{folder_name}")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False,
            pmc_prob = args.pmc_prob,
            pmc_threshold = args.pmc_threshold,
        ) 
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            shuffle=args.shuffle,
        )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args,)
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

        dataset_test = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform, args,)
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    best_val_offset = 1000  # Measured by mean of recall and precision
    last_model_save_path = None

    lr = args.learning_rate 

    train_stats_loss = {lr: []}
    train_stats_loss_offsets = {lr: []}
    train_stats_pose_offsets = {lr: []}
    val_stats_pose_offsets = {lr: []}
    test_stats_pose_offsets = {lr: []}

    model = CrossMatch(
        dataset_train.get_known_classes(),
        COLOR_NAMES_K360,
        args,
    )
    if bool(args.continue_path):
        model_dic = torch.load(args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict = False)

    model.to(device)

    criterion_offsets = nn.MSELoss()

    # Warm-up
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)

    num_epoch_warmup = 3
    for epoch in range(1, args.epochs + 1):
        if epoch == num_epoch_warmup:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if args.lr_scheduler == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
            elif args.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
            else:
                raise TypeError

        train_out = train_epoch(model, dataloader_train, args,)

        train_stats_loss[lr].append(train_out.loss)
        train_stats_loss_offsets[lr].append(train_out.loss_offsets)
        train_stats_pose_offsets[lr].append(train_out.pose_offsets)

        val_out = eval_epoch(model, dataloader_val, args,)  # CARE: which loader for val!
        val_stats_pose_offsets[lr].append(val_out.pose_offsets)

        print()

        test_out = eval_epoch(model, dataloader_test, args,)  # CARE: which loader for test!
        test_stats_pose_offsets[lr].append(test_out.pose_offsets)

        print()

        if scheduler:
            scheduler.step()

        print(
            (
                f"\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} "
                f"t-offset {train_out.pose_offsets:0.3f} "
                f"v-offset {val_out.pose_offsets:0.3f} "
                f"e-offset {test_out.pose_offsets:0.3f} "
            ),
            flush=True,
        )

        offset = np.mean(val_out.pose_offsets)
        if offset < best_val_offset:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/fine_cont{cont}_epoch{epoch}_offset{offset:0.3f}_lr{args.learning_rate}_obj-{args.num_mentioned}-{args.pad_size}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print("Saving model to", model_path)
            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path is not None
                    and last_model_save_path != model_path
                    and osp.isfile(last_model_save_path)
                ):
                    print("Removing", last_model_save_path)
                    os.remove(last_model_save_path)
                last_model_save_path = model_path
            except Exception as e:
                print("Error saving model!", str(e))
            best_val_offset = offset
