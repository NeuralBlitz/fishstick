"""
Pose Visualization Utilities

Visualization tools for pose estimation including skeleton drawing,
heatmap visualization, and pose animation.
"""

from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

import torch
from torch import Tensor
import numpy as np

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


COCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

COCO_KEYPOINT_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

HAND_SKELETON = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class PoseVisualizer:
    """
    Pose visualization utility class.

    Args:
        skeleton: List of (source, target) index pairs
        keypoint_colors: List of RGB colors for each keypoint
        line_color: Color for skeleton lines
        line_thickness: Thickness of skeleton lines
    """

    skeleton: List[Tuple[int, int]] = None
    keypoint_colors: List[Tuple[int, int, int]] = None
    line_color: Tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2

    def __post_init__(self):
        if self.skeleton is None:
            self.skeleton = COCO_SKELETON
        if self.keypoint_colors is None:
            self.keypoint_colors = COCO_KEYPOINT_COLORS


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    radius: int = 4,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw keypoints on image.

    Args:
        image: Image array (H, W, 3)
        keypoints: Keypoint coordinates (N, 2) or (N, 3)
        visibility: Optional visibility flags (N,)
        colors: Optional colors for each keypoint
        radius: Keypoint radius
        conf_threshold: Confidence threshold

    Returns:
        Image with drawn keypoints
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    if colors is None:
        colors = COCO_KEYPOINT_COLORS

    output = image.copy()

    if keypoints.shape[1] >= 3:
        confidences = keypoints[:, 2]
    else:
        confidences = np.ones(keypoints.shape[0])

    for i, (x, y) in enumerate(keypoints[:, :2]):
        if visibility is not None and visibility[i] == 0:
            continue

        if confidences[i] < conf_threshold:
            continue

        color = colors[i % len(colors)]

        cv2.circle(output, (int(x), int(y)), radius, color, -1)

    return output


def draw_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    skeleton: List[Tuple[int, int]] = None,
    visibility: Optional[np.ndarray] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_color: Tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 2,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """
    Draw skeleton on image.

    Args:
        image: Image array (H, W, 3)
        keypoints: Keypoint coordinates (N, 2) or (N, 3)
        skeleton: List of (source, target) pairs
        visibility: Optional visibility flags
        colors: Optional colors for keypoints
        line_color: Color for skeleton lines
        line_thickness: Thickness of lines
        conf_threshold: Confidence threshold

    Returns:
        Image with drawn skeleton
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    if skeleton is None:
        skeleton = COCO_SKELETON

    if colors is None:
        colors = COCO_KEYPOINT_COLORS

    output = draw_keypoints(
        image, keypoints, visibility, colors, conf_threshold=conf_threshold
    )

    if keypoints.shape[1] >= 3:
        confidences = keypoints[:, 2]
    else:
        confidences = np.ones(keypoints.shape[0])

    for src_idx, dst_idx in skeleton:
        if src_idx >= keypoints.shape[0] or dst_idx >= keypoints.shape[0]:
            continue

        if visibility is not None and (
            visibility[src_idx] == 0 or visibility[dst_idx] == 0
        ):
            continue

        if (
            confidences[src_idx] < conf_threshold
            or confidences[dst_idx] < conf_threshold
        ):
            continue

        pt1 = (int(keypoints[src_idx, 0]), int(keypoints[src_idx, 1]))
        pt2 = (int(keypoints[dst_idx, 0]), int(keypoints[dst_idx, 1]))

        cv2.line(output, pt1, pt2, line_color, line_thickness)

    return output


def draw_heatmap(
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    normalize: bool = True,
) -> np.ndarray:
    """
    Draw heatmap as colored image.

    Args:
        heatmap: Heatmap array (H, W)
        colormap: OpenCV colormap
        normalize: Whether to normalize values

    Returns:
        Colored heatmap image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    heatmap = (heatmap * 255).astype(np.uint8)

    colored = cv2.applyColorMap(heatmap, colormap)

    return colored


def draw_paf(
    paf: np.ndarray,
    threshold: float = 0.1,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Draw Part Affinity Fields.

    Args:
        paf: PAF array (H, W, 2) or (2, H, W)
        threshold: Threshold for visualization
        colormap: OpenCV colormap

    Returns:
        PAF visualization
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    if paf.shape[0] == 2:
        paf = paf.transpose(1, 2, 0)

    magnitude = np.sqrt(paf[:, :, 0] ** 2 + paf[:, :, 1] ** 2)

    angle = np.arctan2(paf[:, :, 1], paf[:, :, 0])

    vis = np.zeros((*magnitude.shape, 3), dtype=np.uint8)

    hsv = np.zeros((*magnitude.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude / (threshold + 1e-8) * 255, 0, 255).astype(np.uint8)

    vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return vis


def create_pose_animation(
    poses: List[np.ndarray],
    image_size: Tuple[int, int] = (512, 512),
    skeleton: Optional[List[Tuple[int, int]]] = None,
    fps: int = 30,
    duration: float = 5.0,
) -> np.ndarray:
    """
    Create animation from pose sequence.

    Args:
        poses: List of pose arrays (N, 2) or (N, 3)
        image_size: Size of output frames
        skeleton: Optional skeleton connectivity
        fps: Frames per second
        duration: Animation duration in seconds

    Returns:
        Animation frames (T, H, W, 3)
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    num_frames = int(fps * duration)

    frames = np.zeros((num_frames, *image_size, 3), dtype=np.uint8)

    num_poses = len(poses)

    for i in range(num_frames):
        pose_idx = int(i / num_frames * num_poses)

        frame = np.ones((*image_size, 3), dtype=np.uint8) * 255

        pose = poses[min(pose_idx, num_poses - 1)]

        if pose.shape[1] >= 2:
            scale_x = image_size[1] / 512.0
            scale_y = image_size[0] / 512.0

            scaled_pose = pose.copy()
            scaled_pose[:, 0] *= scale_x
            scaled_pose[:, 1] *= scale_y

            frame = draw_skeleton(
                frame,
                scaled_pose,
                skeleton,
                line_color=(100, 100, 100),
                line_thickness=3,
            )

        frames[i] = frame

    return frames


def plot_pose_statistics(
    keypoints_history: List[np.ndarray],
    title: str = "Pose Statistics",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Plot pose statistics over time.

    Args:
        keypoints_history: List of keypoint arrays
        title: Plot title
        save_path: Optional path to save plot

    Returns:
        Plot image
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for statistics plotting")

    num_frames = len(keypoints_history)

    if num_frames == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title)

    x_positions = [kp[:, 0].mean() for kp in keypoints_history if len(kp) > 0]
    y_positions = [kp[:, 1].mean() for kp in keypoints_history if len(kp) > 0]

    axes[0, 0].plot(x_positions)
    axes[0, 0].set_title("X Position Over Time")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("X")

    axes[0, 1].plot(y_positions)
    axes[0, 1].set_title("Y Position Over Time")
    axes[0, 1].set_xlabel("Frame")
    axes[0, 1].set_ylabel("Y")

    movements = []
    for i in range(1, len(keypoints_history)):
        if len(keypoints_history[i]) > 0 and len(keypoints_history[i - 1]) > 0:
            move = np.linalg.norm(
                keypoints_history[i][:, :2].mean(axis=0)
                - keypoints_history[i - 1][:, :2].mean(axis=0)
            )
            movements.append(move)

    axes[1, 0].plot(movements)
    axes[1, 0].set_title("Movement Magnitude")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("Distance")

    confidences = []
    for kp in keypoints_history:
        if len(kp) > 0 and kp.shape[1] >= 3:
            confidences.append(kp[:, 2].mean())
        else:
            confidences.append(1.0)

    axes[1, 1].plot(confidences)
    axes[1, 1].set_title("Confidence Over Time")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("Confidence")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return data


def visualize_multipose(
    image: np.ndarray,
    poses: List[Dict[str, Any]],
    conf_threshold: float = 0.3,
    color_map: str = "rainbow",
) -> np.ndarray:
    """
    Visualize multiple poses on image.

    Args:
        image: Input image
        poses: List of pose dictionaries with 'keypoints', 'scores'
        conf_threshold: Confidence threshold
        color_map: Color mapping strategy

    Returns:
        Visualization image
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV is required for visualization")

    output = image.copy()

    num_poses = len(poses)

    for i, pose in enumerate(poses):
        keypoints = pose.get("keypoints", pose.get("keypoints_2d"))

        if keypoints is None:
            continue

        hue = int(180 * i / max(num_poses, 1))
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]

        output = draw_skeleton(
            output,
            keypoints,
            COCO_SKELETON,
            line_color=tuple(int(c) for c in color),
            line_thickness=2,
            conf_threshold=conf_threshold,
        )

        bbox = pose.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return output


__all__ = [
    "PoseVisualizer",
    "draw_keypoints",
    "draw_skeleton",
    "draw_heatmap",
    "draw_paf",
    "create_pose_animation",
    "plot_pose_statistics",
    "visualize_multipose",
    "COCO_SKELETON",
    "HAND_SKELETON",
    "COCO_KEYPOINT_COLORS",
]
