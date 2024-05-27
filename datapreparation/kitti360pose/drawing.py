"""Various functions for plotting or interactively showing cells, poses and scenes.
"""

from typing import List
import numpy as np
import cv2
from datapreparation.kitti360pose.imports import (
    Object3d,
    Cell,
    Pose,
    DescriptionPoseCell,
    DescriptionBestCell,
)
from datapreparation.kitti360pose.utils import CLASS_TO_COLOR

try:
    import pptk
except:
    pass


def show_pptk(xyz, rgb):
    viewer = pptk.viewer(xyz)
    if isinstance(rgb, np.ndarray):
        viewer.attributes(rgb.astype(np.float32))
    else:
        attributes = [x.astype(np.float32) for x in rgb]
        viewer.attributes(*attributes)

    viewer.set(point_size=0.1)

    return viewer


# Use scale=100 for cell-objects
def show_objects(objects: List[Object3d], scale=1.0):
    num_points = np.sum([len(obj.xyz) for obj in objects])
    xyz = np.zeros((num_points, 3), dtype=np.float)
    rgb1 = np.zeros((num_points, 3), dtype=np.uint8)
    rgb2 = np.zeros((num_points, 3), dtype=np.uint8)
    rgb3 = np.zeros((num_points, 3), dtype=np.uint8)
    offset = 0
    for obj in objects:
        rand_color = np.random.randint(low=0, high=256, size=3)
        c = CLASS_TO_COLOR[obj.label]
        # xyz = np.vstack((xyz, obj.xyz))
        # rgb1 = np.vstack((rgb1, np.ones((len(obj.xyz), 3))*rand_color ))
        # rgb2 = np.vstack((rgb2, obj.rgb))
        # rgb3 = np.vstack((rgb3, np.ones((len(obj.xyz), 3))*np.array(c) ))
        xyz[offset : offset + len(obj.xyz)] = obj.xyz
        rgb1[offset : offset + len(obj.xyz)] = np.ones((len(obj.xyz), 3)) * rand_color
        rgb2[offset : offset + len(obj.xyz)] = obj.rgb * 255
        rgb3[offset : offset + len(obj.xyz)] = np.ones((len(obj.xyz), 3)) * np.array(c)
        offset += len(obj.xyz)
    return show_pptk(xyz * scale, [rgb1 / 255.0, rgb2 / 255.0, rgb3 / 255.0])


def plot_objects(objects, pose=None, scale=1024):
    img = np.zeros((scale, scale, 3), dtype=np.uint8)
    for obj in objects:
        c = CLASS_TO_COLOR[obj.label]
        for point in obj.xyz:
            point = np.int0((point[0:2] + 0.5) * scale / 2)
            cv2.circle(img, tuple(point), 1, (int(c[2]), int(c[1]), int(c[0])))
    if pose is not None:
        point = np.int0((pose[0:2] + 0.5) * scale / 2)
        cv2.circle(img, tuple(point), scale // 50, (255, 0, 255))

    return cv2.flip(img, 0)  # Flip for correct north/south


def plot_cell(
    cell: Cell,
    scale=1024,
    use_rgb=False,
    use_instances=False,
    point_size=6,
    pose: np.ndarray = None,
):
    img = np.ones((scale, scale, 3), dtype=np.uint8) * 255
    # Draw points of each object
    for obj in cell.objects:
        if obj.label == "pad":
            continue
        c = np.random.randint(256, size=3) if use_instances else CLASS_TO_COLOR[obj.label]
        for i_point, point in enumerate(obj.xyz * scale):
            if use_rgb:
                c = tuple(np.uint8(obj.rgb[i_point] * 255))
            point = np.int0(point[0:2])
            cv2.circle(
                img, tuple(point), point_size, (int(c[2]), int(c[1]), int(c[0])), thickness=-1
            )

    if pose is not None:
        point = np.int0(pose[0:2] * scale)
        cv2.circle(img, tuple(point), 40, (0, 0, 255), thickness=-1)

    return cv2.flip(img, 0)  # Flip for correct north/south


def plot_matches_in_best_cell(
    cell: Cell, pose: Pose, pred_matches, gt_matches, scale=1024, point_size=6
):
    """pred_matches as [hint_idx, hint_idx, hint_idx...], gt_matches as [(obj_idx, hint_idx), (...)]"""
    assert cell.id == pose.cell_id
    img = np.ones((scale, scale, 3), dtype=np.uint8) * 255
    cell_objects = cell.objects[0 : len(pred_matches)]  # Cut-off after pad_size

    # Draw instances
    for i_obj, obj in enumerate(cell_objects):
        if obj.label == "pad":
            continue
        color = (
            (128, 128, 128) if pred_matches[i_obj] < 0 else CLASS_TO_COLOR[obj.label]
        )  # Color gray if not matched
        for i_point, point in enumerate(obj.xyz * scale):
            point = np.int0(point[0:2])
            cv2.circle(
                img,
                tuple(point),
                point_size,
                (int(color[2]), int(color[1]), int(color[0])),
                thickness=-1,
            )

    # Draw pose
    point = np.int0(pose.pose[0:2] * scale)
    cv2.circle(img, tuple(point), 30, (0, 0, 255), thickness=-1)

    # Draw arrows from predictions
    for i_obj, obj in enumerate(cell_objects):
        # Skip unmatched objects (based on prediction)
        if pred_matches[i_obj] < 0:
            continue

        obj_idx = i_obj
        hint_idx = pred_matches[i_obj]
        color = (
            (0, 255, 0) if (obj_idx, hint_idx) in gt_matches else (255, 0, 0)
        )  # Green if correct match, red otherwise
        target = np.int0(obj.get_closest_point(pose.pose) * scale)[0:2]
        cv2.arrowedLine(
            img,
            tuple(point),
            tuple(target),
            (int(color[2]), int(color[1]), int(color[0])),
            thickness=7,
        )

    # Draw arrows from missed predictions
    gt_matches_dict = {obj_idx: hint_idx for obj_idx, hint_idx in gt_matches}
    for obj_idx, obj in enumerate(cell_objects):
        if obj_idx not in gt_matches_dict:
            continue
        hint_idx = gt_matches_dict[obj_idx]
        if pred_matches[obj_idx] != hint_idx:
            target = np.int0(obj.get_closest_point(pose.pose) * scale)[0:2]
            cv2.arrowedLine(img, tuple(point), tuple(target), (0, 255, 255), thickness=6)

    return cv2.flip(img, 0)


def depr_plot_matches_in_best_cell(
    cell: Cell,
    pose: Pose,
    true_matches=[],
    false_positives=[],
    false_negatives=[],
    scale=1024,
    point_size=6,
):
    assert cell.id == pose.cell_id
    img = np.ones((scale, scale, 3), dtype=np.uint8) * 255
    # Draw points of each object
    for i_obj, obj in enumerate(cell.objects):
        if obj.label == "pad":
            continue
        if i_obj in true_matches:
            c = (0, 255, 0)
        elif i_obj in false_positives:
            c = (255, 255, 0)
        elif i_obj in false_negatives:
            c = (255, 0, 0)
        else:
            c = (128, 128, 128)
        for i_point, point in enumerate(obj.xyz * scale):
            point = np.int0(point[0:2])
            cv2.circle(
                img, tuple(point), point_size, (int(c[2]), int(c[1]), int(c[0])), thickness=-1
            )
    # Draw pose
    point = np.int0(pose.pose[0:2] * scale)
    cv2.circle(img, tuple(point), 20, (0, 0, 255), thickness=7)
    # Draw arrows
    for i_obj, obj in enumerate(cell.objects):
        if i_obj in true_matches:
            target = np.int0(obj.get_closest_point(pose.pose) * scale)[0:2]
            cv2.arrowedLine(img, tuple(point), tuple(target), (0, 0, 255), thickness=6)
    return cv2.flip(img, 0)


def plot_pose_in_best_cell(cell: Cell, pose: Pose, scale=1024, use_rgb=False, show_unmatched=False):
    img = np.zeros((scale, scale, 3), dtype=np.uint8)
    # Draw points of each object
    for obj in cell.objects:
        if obj.label == "pad":
            continue
        c = CLASS_TO_COLOR[obj.label]
        for i_point, point in enumerate(obj.xyz * scale):
            if use_rgb:
                c = tuple(np.uint8(obj.rgb[i_point] * 255))
            point = np.int0(point[0:2])
            cv2.circle(img, tuple(point), 1, (int(c[2]), int(c[1]), int(c[0])))
    # Draw pose
    point = np.int0(pose.pose[0:2] * scale)
    cv2.circle(img, tuple(point), 10, (0, 0, 255), thickness=3)
    # Draw lines to closest points
    for descr in pose.descriptions:
        if not descr.is_matched and not show_unmatched:
            continue

        target = np.int0(descr.closest_point[0:2] * scale)
        cv2.arrowedLine(img, tuple(point), tuple(target), (0, 0, 255), thickness=2)

    img = cv2.flip(img, 0)
    if not show_unmatched:
        num_unmatched = len([d for d in pose.descriptions if not d.is_matched])
        cv2.putText(
            img,
            f"Unmatched: {num_unmatched}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
        )
    return img


def plot_cells_and_poses(cells: List[Cell], poses: List[Pose], size=1024):
    best_cell_ids = [pose.cell_id for pose in poses]
    pose_locations = np.array([pose.pose_w for pose in poses])
    min_x, max_x, min_y, max_y = (
        np.min(pose_locations[:, 0]),
        np.max(pose_locations[:, 0]),
        np.min(pose_locations[:, 1]),
        np.max(pose_locations[:, 1]),
    )
    scale = max(max_x - min_x, max_y - min_y)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for cell in cells:
        bbox = cell.bbox_w - np.array((min_x, min_y, 0, min_x, min_y, 0))
        bbox = np.int0(bbox / scale * size)
        color = (255, 255, 255) if cell.id in best_cell_ids else (128, 128, 128)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[3], bbox[4]), color, thickness=2)
    for pose in poses:
        p = pose.pose_w - np.array((min_x, min_y, 0))
        p = np.int0(p / scale * size)
        cv2.circle(img, (p[0], p[1]), 6, (0, 0, 255), thickness=4)
    img = cv2.flip(img, 0)
    return img
