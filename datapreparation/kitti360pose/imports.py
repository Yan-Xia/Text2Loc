from typing import List, Union
import numpy as np
import cv2

from datapreparation.kitti360pose.utils import COLORS, COLOR_NAMES


class Object3d:
    # NOTE: use cell-id and scene-names to unique identify objects if needed
    def __init__(self, id: int, instance_id: int, xyz: np.ndarray, rgb: np.ndarray, label: str):
        """Representation of a 3D object in the scene

        Args:
            id (int): Object ID, unique only inside a single cell. Multiple ids can belong to the same instance ID.
            instance_id (int): Original instance ID, can repeat across cells and in the same cell due to clustering of stuff objects.
            xyz (np.ndarray): XYZ coordinates of the object's points
            rgb (np.ndarray): RGB color coordinates of the object's points
            label (str): Class label
        """
        self.id = id
        self.instance_id = instance_id
        self.xyz = xyz
        self.rgb = rgb
        self.label = label
        # self.closest_point = None # Set in get_closest_point() for cell-object. CARE: may now be "incorrect" since multiple poses can use this object/cells
        # self.center = None # TODO, for SG-Matching: ok to just input center instead of closest-point? or better to directly input xyz (care that PN++ knowns about coords)

    def get_color_rgb(self):
        color = np.mean(self.rgb, axis=0)
        assert color.shape == (3,)
        return color

    def get_color_text(self):
        """Get the color as text based on the closest (L2) discrete color-center.
        CARE: Can change during downsampling or masking
        """
        dists = np.linalg.norm(np.mean(self.rgb, axis=0) - COLORS, axis=1)
        return COLOR_NAMES[np.argmin(dists)]

    def get_center(self):
        return np.mean(self.xyz, axis=0)

    def __repr__(self):
        return f"Object3d: {self.label}"

    def apply_downsampling(self, indices):
        self.xyz = self.xyz[indices]
        self.rgb = self.rgb[indices]

    def mask_points(self, mask):
        """Mask xyz and rgb, the id is retained"""
        assert len(mask) > 6  # To prevent bbox input
        # return Object3d(self.xyz[mask], self.rgb[mask], self.label, self.id)
        return Object3d(self.id, self.instance_id, self.xyz[mask], self.rgb[mask], self.label)

    def get_closest_point(self, anchor):
        dists = np.linalg.norm(self.xyz - anchor, axis=1)
        # self.closest_point = self.xyz[np.argmin(dists)]
        return self.xyz[np.argmin(dists)]

    @classmethod
    def merge(cls, obj1, obj2):
        assert (
            obj1.label == obj2.label and obj1.id == obj2.id
        ), f"{obj1.label}, {obj2.label}, {obj1.id}, {obj2.id}"
        return Object3d(
            obj1.id,
            obj1.instance_id,
            np.vstack((obj1.xyz, obj2.xyz)),
            np.vstack((obj1.rgb, obj2.rgb)),
            obj1.label,
        )

    @classmethod
    def create_padding(cls):
        """Create a padding object.

        Returns:
            Object3d: Padding object
        """
        obj = Object3d(-1, -1, np.random.rand(8, 3) * 0.001, np.zeros((8, 3)), "pad")
        obj.get_closest_point([-1, -1, -1])
        return obj


class DescriptionPoseCell:
    def __init__(
        self,
        object: Object3d,
        direction: str,
        offset_center: np.ndarray,
        offset_closest: np.ndarray,
        closest_point: np.ndarray,
    ):
        """Struct for a single description of a pose using an object in context of the "Pose Cell".

        Args:
            object (Object3d): Object used for description
            direction (str): Direction string
            offset_center (np.ndarray): Direction vector from the center of the object to the pose
            offset_closest (np.ndarray): Direction vector from the point of the object that is closest to the pose to the pose
            closest_point (np.ndarray): Coordinates of the object's point that is closest to the pose.
        """
        self.object_id = object.id
        self.object_instance_id = object.instance_id
        self.object_label = object.label
        self.object_color_rgb = object.get_color_rgb()
        self.object_color_text = object.get_color_text()
        self.direction = direction  # Text might not match offset later in best-cell!
        self.offset_center = offset_center[0:2]  # Offset to center of object
        self.offset_closest = offset_closest[0:2]  # Offset to closest-point of object
        self.closest_point = closest_point[0:2]  # Only in pose-cell!

    def __repr__(self):
        return f"Pose is {self.direction} of a {self.object_color_text} {self.object_label}"


# TODO/CARE: Match on offset_closest (less likely to change) but train on offset_center (relevant in evaluation)
class DescriptionBestCell:
    """Struct for a single description of a pose using an object in context of the "Best Cell".
    The "Best Cell" is that cell of the dataset that is closest to the pose.

    Current offset policy: all taken from pose_cell and passed through. Training on center_offsets
    """

    @classmethod
    def from_matched(
        cls,
        descr: DescriptionPoseCell,
        object_id,
        best_closest_point,
        best_offset_center,
        best_offset_closest,
    ):
        d = DescriptionBestCell()
        # Original attributes
        d.object_instance_id = descr.object_instance_id
        d.object_label = descr.object_label
        d.object_color_rgb = descr.object_color_rgb
        d.object_color_text = descr.object_color_text
        d.direction = descr.direction
        d.offset_center = descr.offset_center  # Retained from pose-cell
        d.offset_closest = descr.offset_closest  # Retained from pose-cell

        # Updated attributes matched in best-cell
        d.object_id = object_id
        # d.offset_center = offset_center[0:2]
        # d.offset_closest = offset_closest[0:2]
        d.closest_point = best_closest_point[0:2]  # Updated to best-cell
        d.best_offset_center = best_offset_center[0:2]  # Updated to best-cell
        d.best_offset_closest = best_offset_closest[0:2]  # Updated to best-cell
        d.is_matched = True
        return d

    @classmethod
    def from_unmatched(cls, descr: DescriptionPoseCell):
        d = DescriptionBestCell()
        # Original attributes
        d.object_instance_id = descr.object_instance_id
        d.object_label = descr.object_label
        d.object_color_rgb = descr.object_color_rgb
        d.object_color_text = descr.object_color_text
        d.direction = descr.direction
        d.offset_center = descr.offset_center
        d.offset_closest = descr.offset_closest

        d.closest_point = descr.closest_point  # Only for debug!

        d.is_matched = False
        return d

    def __repr__(self):
        return f"Pose is {self.direction} of a {self.object_color_text} {self.object_label}" + (
            " (✓)" if self.is_matched else " (☓)"
        )


class Pose:
    def __init__(
        self,
        pose_in_cell: np.ndarray,
        pose_w: np.ndarray,
        cell_id: int,
        scene_name: str,
        descriptions: List[DescriptionBestCell],
        described_by: str = None,
    ):
        """Struct of a single query pose.

        Args:
            pose_in_cell (np.ndarray): Pose coordinates normed inside the "Best Cell" ∈ [0, 1]
            pose_w (np.ndarray): Pose world coordinates
            cell_id (int): ID of the Best Cell
            scene_name (str): Scene name
            descriptions (List[DescriptionBestCell]): Description of the pose in context of the Best Cell
            described_by (str, optional): Defaults to None.
        """
        assert isinstance(descriptions[0], DescriptionBestCell)
        self.pose = (
            pose_in_cell  # The pose in the best cell (specified by cell_id), normed to ∈ [0, 1]
        )
        self.pose_w = pose_w
        self.cell_id = cell_id  # ID of the best cell in the database
        self.descriptions = descriptions
        self.scene_name = scene_name
        self.described_by = described_by

    def __repr__(self) -> str:
        return f"Pose at {self.pose_w} in {self.cell_id}"

    def get_text(self):
        text = ""
        for d in self.descriptions:
            text += str(d) + ". "
        return text

    def get_number_unmatched(self):
        return len([d for d in self.descriptions if not d.is_matched])


class Cell:
    def __init__(self, idx, scene_name, objects: List[Object3d], cell_size, bbox_w):
        """Instance of a single cell.

        Args:
            IDs should be unique across entire dataset
            Objects include distractors and mentioned, already cropped and normalized in cell
            Pose as (x,y,z), already normalized in cell
            cell-size: longest edge in world-coordinates
            Pose_w as (x,y,z) in original world-coordinates
        """
        self.scene_name = scene_name
        self.id = f"{scene_name}_{idx:05.0f}"  # Incrementing alpha-numeric id in format 00XX_XXXXX
        assert len(self.id) == 10, self.id
        self.objects = objects
        # self.descriptions = descriptions
        # self.pose = pose

        self.cell_size = cell_size  # Original cell-size (longest edge)
        # self.pose_w = pose_w # Original pose in world-coordinates
        self.bbox_w = bbox_w  # Original pose in world-coordinates

    def __repr__(self):
        return f"Cell {self.id}: {len(self.objects)} objects at {np.int0(self.bbox_w)}"

    def get_center(self):
        return 1 / 2 * (self.bbox_w[0:3] + self.bbox_w[3:6])
