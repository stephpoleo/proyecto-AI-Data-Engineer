"""
Utility functions for computer vision operations.
Contains reusable functions for geometry calculations, color analysis, and visualization.
"""

from typing import Tuple

import cv2
import numpy as np

from .config import FONT_COLOR, CLASS_COLORS, DEFAULT_COLOR, OUTPUT_DATA_PATH


def calculate_bbox_attributes(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
) -> dict:
    """
    Calculate all bounding box related attributes.

    Args:
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        frame_shape: Frame dimensions (height, width)

    Returns:
        dict: Dictionary with calculated attributes
    """
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]

    width = x2 - x1
    height = y2 - y1
    area_pixels = width * height

    center_x = x1 + width // 2
    center_y = y1 + height // 2

    center_x_norm = center_x / frame_width if frame_width > 0 else 0
    center_y_norm = center_y / frame_height if frame_height > 0 else 0

    frame_area = frame_width * frame_height
    bbox_area_ratio = area_pixels / frame_area if frame_area > 0 else 0

    position_region = calculate_position_region(center_x_norm, center_y_norm)

    return {
        "x_min": x1,
        "y_min": y1,
        "x_max": x2,
        "y_max": y2,
        "width": width,
        "height": height,
        "area_pixels": area_pixels,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "bbox_area_ratio": bbox_area_ratio,
        "center_x": center_x,
        "center_y": center_y,
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "position_region": position_region,
    }


def calculate_position_region(center_x_norm: float, center_y_norm: float) -> str:
    """
    Calculate position region based on normalized center coordinates.

    Args:
        center_x_norm: Normalized x center (0-1)
        center_y_norm: Normalized y center (0-1)

    Returns:
        str: Position region (e.g., "top-left", "middle-center", etc.)
    """
    if center_y_norm < 0.33:
        vertical = "top"
    elif center_y_norm < 0.67:
        vertical = "middle"
    else:
        vertical = "bottom"

    if center_x_norm < 0.33:
        horizontal = "left"
    elif center_x_norm < 0.67:
        horizontal = "center"
    else:
        horizontal = "right"

    return f"{vertical}-{horizontal}"


def calculate_dominant_color(
    frame: np.ndarray, bbox: Tuple[int, int, int, int]
) -> dict:
    """
    Calculate dominant color from the bounding box region.

    Args:
        frame: Input frame
        bbox: Bounding box coordinates (x1, y1, x2, y2)

    Returns:
        dict: Dictionary with dominant color information
    """
    x1, y1, x2, y2 = bbox

    if x1 >= x2 or y1 >= y2:
        return _get_default_color_dict()

    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return _get_default_color_dict()

    mean_color = np.mean(roi, axis=(0, 1)).astype(int)
    dom_b, dom_g, dom_r = mean_color

    color_name = get_color_name(dom_r, dom_g, dom_b)

    return {
        "dominant_color_name": color_name,
        "dom_r": int(dom_r),
        "dom_g": int(dom_g),
        "dom_b": int(dom_b),
    }


def get_color_name(r: int, g: int, b: int) -> str:
    """
    Determine color name based on RGB values.

    Args:
        r, g, b: RGB color values (0-255)

    Returns:
        str: Color name
    """
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    max_val = max(r, g, b)
    min_val = min(r, g, b)

    if max_val - min_val < 30:
        if max_val < 50:
            return "black"
        elif max_val > 200:
            return "white"
        else:
            return "gray"

    if r > g and r > b:
        if r > 150:
            return "red"
    elif g > r and g > b:
        if g > 150:
            return "green"
    elif b > r and b > g:
        if b > 150:
            return "blue"

    if r > 150 and g > 150 and b < 100:
        return "yellow"
    elif r > 150 and b > 150 and g < 100:
        return "magenta"
    elif g > 150 and b > 150 and r < 100:
        return "cyan"

    return "mixed"


def get_class_color(class_name: str) -> Tuple[int, int, int]:
    """
    Get color for a specific class name.

    Args:
        class_name: Name of the detected class

    Returns:
        Tuple[int, int, int]: BGR color tuple
    """
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)


def draw_multiple_detections(
    frame: np.ndarray,
    detections: list,
    font_color: Tuple[int, int, int] = FONT_COLOR,
) -> None:
    """
    Draw multiple detection bounding boxes and labels on frame with class-specific colors.

    Args:
        frame: Frame to draw on (modified in-place)
        detections: List of detections [(class_name, confidence, bbox, class_id), ...]
        font_color: Color for text (BGR format)
    """
    for class_name, confidence, bbox, _ in detections:
        class_color = get_class_color(class_name)
        draw_detection_box(frame, bbox, class_name, confidence, class_color, font_color)


def draw_detection_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    box_color: Tuple[int, int, int] = None,
    font_color: Tuple[int, int, int] = FONT_COLOR,
) -> None:
    """
    Draw detection bounding box and label on frame with class-specific color.

    Args:
        frame: Frame to draw on (modified in-place)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        class_name: Detected class name
        confidence: Detection confidence score
        box_color: Color for bounding box (BGR format). If None, uses class-specific color
        font_color: Color for text (BGR format)
    """
    if box_color is None:
        box_color = get_class_color(class_name)
    x1, y1, x2, y2 = bbox

    frame_height, frame_width = frame.shape[:2]
    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

    label = f"{class_name} {confidence:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)

    text_y = max(text_height + 10, y1)
    text_x = min(x1, frame_width - text_width)

    cv2.rectangle(
        frame,
        (text_x, text_y - text_height - 10),
        (text_x + text_width, text_y),
        box_color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (text_x, text_y - 5),
        font,
        font_scale,
        font_color,
        thickness,
    )


def extract_filename_from_path(file_path: str) -> str:
    """
    Extract filename from a file path (cross-platform).

    Args:
        file_path: Full path to file

    Returns:
        str: Filename without path
    """
    if "\\" in file_path:
        return file_path.split("\\")[-1]
    elif "/" in file_path:
        return file_path.split("/")[-1]
    else:
        return file_path


def validate_bbox_coordinates(
    bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Validate and clamp bounding box coordinates to frame boundaries.

    Args:
        bbox: Original bounding box (x1, y1, x2, y2)
        frame_shape: Frame dimensions (height, width)

    Returns:
        Tuple[int, int, int, int]: Validated bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame_shape[:2]

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))
    x2 = max(x1 + 1, min(x2, frame_width))
    y2 = max(y1 + 1, min(y2, frame_height))

    return (x1, y1, x2, y2)


def _get_default_color_dict() -> dict:
    """
    Get default color dictionary for invalid regions.

    Returns:
        dict: Default color information
    """
    return {
        "dominant_color_name": "unknown",
        "dom_r": 0,
        "dom_g": 0,
        "dom_b": 0,
    }


def create_detection_dataframe_schema() -> list:
    """
    Get the schema (column names) for the detection dataframe.

    Returns:
        list: List of column names for the detection dataframe
    """
    return [
        "source_type",
        "source_id",
        "frame_number",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
        "width",
        "height",
        "area_pixels",
        "frame_width",
        "frame_height",
        "bbox_area_ratio",
        "center_x",
        "center_y",
        "center_x_norm",
        "center_y_norm",
        "position_region",
        "dominant_color_name",
        "dom_r",
        "dom_g",
        "dom_b",
        "timestamp_sec",
    ]


def save_dataframe_to_csv(dataframe, filename) -> None:
    """
    Save detection dataframe to CSV file.

    Args:
        dataframe: pandas DataFrame with detection data
        filename: Name of CSV file to save
    """
    path = OUTPUT_DATA_PATH + filename
    if not dataframe.empty:
        dataframe.to_csv(path, index=False)
        print(f"Detection data saved to {path}")
        print(f"Total detections: {len(dataframe)}")
    else:
        print("No detection data to save")


def get_detection_summary(dataframe) -> dict:
    """
    Get summary statistics from detection dataframe.

    Args:
        dataframe: pandas DataFrame with detection data

    Returns:
        dict: Summary statistics
    """
    if dataframe.empty:
        return {"total_detections": 0}

    summary = {
        "total_detections": len(dataframe),
        "unique_classes": dataframe["class_name"].nunique(),
        "classes_detected": dataframe["class_name"].unique().tolist(),
        "source_types": dataframe["source_type"].unique().tolist(),
        "avg_confidence": dataframe["confidence"].mean(),
        "frame_range": [
            dataframe["frame_number"].min(),
            dataframe["frame_number"].max(),
        ],
    }

    return summary


def add_detection_to_dataframe(
    dataframe,
    source_type: str,
    source_id: str,
    frame,
    class_name: str,
    confidence: float,
    bbox: Tuple[int, int, int, int],
    class_id: int,
    frame_counter: int,
    start_time: float,
) -> None:
    """
    Add detection data to dataframe.

    Args:
        dataframe: pandas DataFrame to append to
        source_type: Type of source ("image", "video", "camera")
        source_id: Source identifier
        frame: Current frame (numpy array)
        class_name: Detected class name
        confidence: Detection confidence
        bbox: Bounding box coordinates
        class_id: Class ID number
        frame_counter: Current frame number
        start_time: Processing start time

    Returns:
        Updated dataframe
    """
    import time
    import pandas as pd

    bbox_attrs = calculate_bbox_attributes(bbox, frame.shape)
    color_attrs = calculate_dominant_color(frame, bbox)

    timestamp_sec = time.time() - start_time

    new_row = {
        "source_type": source_type,
        "source_id": source_id,
        "frame_number": frame_counter,
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "timestamp_sec": timestamp_sec,
        **bbox_attrs,
        **color_attrs,
    }

    return pd.concat([dataframe, pd.DataFrame([new_row])], ignore_index=True)
