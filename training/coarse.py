"""Module for training the coarse cell-retrieval module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
import collections

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp
import tqdm

from models.cell_retrieval import CellRetrievalNetwork

from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.utils import COLOR_NAMES as COLOR_NAMES_K360
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ContrastiveLoss
from training.utils import plot_retrievals

"Training Process for global place recognition"
def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))


    for i_batch, batch in pbar:

        optimizer.zero_grad()

        anchor = model.encode_text(batch["texts"])
        positive = model.encode_objects(batch["objects"], batch["object_points"])

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        else:
            loss = criterion(anchor, positive)

        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        torch.cuda.empty_cache()

    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False):
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    cell_encodings = np.zeros((len(cells_dataset), model.embed_dim))
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = np.zeros((len(dataloader.dataset), model.embed_dim))
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0
    for batch in tqdm.tqdm(dataloader):

        text_enc = model.encode_text(batch["texts"])
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = (
            text_enc.cpu().detach().numpy()
        )
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in cells_dataloader:
        cell_enc = model.encode_objects(batch["objects"], batch["object_points"])
        batch_size = len(cell_enc)

        cell_encodings[index_offset : index_offset + batch_size, :] = (
            cell_enc.cpu().detach().numpy()
        )
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []
    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            scores = cell_encodings[:] @ text_encodings[query_idx]
            assert len(scores) == len(dataloader.dataset.all_cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    elif return_distance:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings, np.stack(dists_list), np.stack(scores_list)
    else:
        return accuracies, accuracies_close, top_retrievals


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
        # ['2013_05_28_drive_0003_sync', ]
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

        dataset_train = Kitti360CoarseDatasetMulti(
            args.base_path,
            SCENE_NAMES_TRAIN,
            train_transform,
            shuffle_hints=True,
            flip_poses=True,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=args.shuffle,
            num_workers=args.cpus,
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform,)

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

        dataset_test = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform,)

        dataloader_test = DataLoader(
            dataset_test,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]
    assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    lr = args.learning_rate

    dict_loss = {lr: []}
    dict_acc = {k: [] for k in args.top_k}
    dict_acc_val = {k: [] for k in args.top_k}
    dict_acc_val_close = {k: [] for k in args.top_k}
    dict_acc_test = {k: [] for k in args.top_k}
    dict_acc_test_close = {k: [] for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path_val = None

    model = CellRetrievalNetwork(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args,
        )
    if args.continue_path:
        model_dic = torch.load(args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict = False)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.ranking_loss == "pairwise":
        criterion = PairwiseRankingLoss(margin=args.margin)
    if args.ranking_loss == "hardest":
        criterion = HardestRankingLoss(margin=args.margin)
    if args.ranking_loss == "triplet":
        criterion = nn.TripletMarginLoss(margin=args.margin)
    if args.ranking_loss == "contrastive":
        criterion = ContrastiveLoss(temperature=args.temperature)

    if args.lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    else:
        raise TypeError

    for epoch in range(1, args.epochs + 1):
        # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
        
        loss, train_batches = train_epoch(model, dataloader_train, args)
        train_acc, train_acc_close, train_retrievals = eval_epoch(
            model, dataloader_train, args
        )  
        val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
        test_acc, test_acc_close, test_retrievals = eval_epoch(model, dataloader_test, args)

        key = lr
        dict_loss[key].append(loss)
        for k in args.top_k:
            dict_acc[k].append(train_acc[k])
            dict_acc_val[k].append(val_acc[k])
            dict_acc_val_close[k].append(val_acc_close[k])
            dict_acc_test[k].append(test_acc[k])
            dict_acc_test_close[k].append(test_acc_close[k])

        scheduler.step()
        print(f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} train-acc: ", end="")
        for k, v in train_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc: ", end="")
        for k, v in val_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc-close: ", end="")
        for k, v in val_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")

        print("test-acc: ", end="")
        for k, v in test_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("test-acc-close: ", end="")
        for k, v in test_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("\n", flush=True)

        # Saving best model
        acc_val = val_acc[max(args.top_k)]
        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/coarse_cont{cont}_epoch{epoch}_acc{acc_val:0.3f}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_loss-{args.ranking_loss}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print(f"Saving model at {acc_val:0.2f} to {model_path}")
            
            try:
                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path_val is not None
                    and last_model_save_path_val != model_path
                    and osp.isfile(last_model_save_path_val)
                ):  
                    print("Removing", last_model_save_path_val)
                    os.remove(last_model_save_path_val)
                
                last_model_save_path_val = model_path
                
            except Exception as e:
                print(f"Error saving model!", str(e))
            best_val_accuracy = acc_val
                           

