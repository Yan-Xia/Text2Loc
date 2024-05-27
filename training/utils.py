import numpy as np
import cv2
from easydict import EasyDict
from numpy.lib.arraysetops import isin
from dataloading.kitti360pose.base import Kitti360BaseDataset
from datapreparation.kitti360pose.drawing import plot_cell, plot_matches_in_best_cell

import torch_geometric.transforms as T


def set_border(img, color):
    img[0:15, :] = color
    img[-15:, :] = color
    img[:, 0:15] = color
    img[:, -15:] = color


def plot_matches(matches0, dataset, count=20):
    print(len(matches0), len(dataset))
    print(matches0[0])
    assert isinstance(matches0, list) and len(matches0) == len(dataset)

    for _ in range(count):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        pose = data["poses"]
        cell = data["cells"]
        pred_matches = matches0[idx]
        gt_matches = data["matches"]

        img = plot_matches_in_best_cell(cell, pose, pred_matches, gt_matches)
        cv2.imwrite(f"matches_{idx}.png", img)
        print(f"Idx {idx} saved!")


def plot_retrievals(top_retrievals, dataset, count=10, top_k=3, green_thresh=10):
    assert isinstance(top_retrievals, list)

    cells_dict = {cell.id: cell for cell in dataset.all_cells}

    count_plotted = 0
    while count_plotted < count:
        idx = np.random.randint(len(top_retrievals))
        pose = dataset.all_poses[idx]
        retrievals = top_retrievals[idx]
        # if pose.cell_id in retrievals[0 : top_k]:

        images_semantic_rgb = []
        for use_rgb in (False, True):
            images = []
            # Add query
            images.append(plot_cell(cells_dict[pose.cell_id], use_rgb=use_rgb, pose=pose.pose))

            # Add top-k
            for cell_id in retrievals[0:top_k]:
                cell = cells_dict[cell_id]
                img = plot_cell(cell, use_rgb=use_rgb)

                # Add border and distance
                dist = np.linalg.norm(pose.pose_w[0:2] - cell.get_center()[0:2])
                border_color = (
                    (0, 255, 0)
                    if pose.scene_name == cell.scene_name and dist < green_thresh
                    else (0, 0, 255)
                )
                set_border(img, border_color)

                h, w, d = img.shape
                cv2.rectangle(
                    img,
                    (25, h - 125),
                    (img.shape[1] // 5 * 3, h - 20),
                    (255, 255, 255),
                    thickness=-1,
                )
                dist_text = "n/a" if pose.scene_name != cell.scene_name else f"{dist:0.2f}"
                cv2.putText(
                    img,
                    "Distance: " + dist_text,
                    (50, img.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 0),
                    thickness=2,
                )

                images.append(img)

            sep = np.ones((images[0].shape[0], 100, 3), np.uint8) * 255
            images.insert(1, sep)
            images_semantic_rgb.append(np.hstack(images))

        cv2.imwrite(f"ret_{count_plotted}.png", np.vstack(images_semantic_rgb))
        print("Saved pos!")
        count_plotted += 1

