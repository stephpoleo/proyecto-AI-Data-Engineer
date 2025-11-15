"""
Module for handling media input/output operations including camera, image, and video processing.
Provides a MediaIO class with methods for camera operations, file I/O, and object detection visualization.
"""

from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import (
    BOX_COLOR,
    CAM_INDEX,
    FONT_COLOR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMG_INPUT_PATH,
    VIDEO_INPUT_PATH,
    WINDOW_TITLE,
)
from .model import YoloModel


class MediaIO:
    """
    Media input/output handler for computer vision operations.

    Provides methods for:
    - Camera operations (open, preview, release)
    - Image file operations (read, preview)
    - Video file operations (read, preview)
    - Object detection visualization
    """

    def __init__(self) -> None:
        """Initialize MediaIO instance."""
        self._yolo_model = YoloModel()
        self.dataframe = pd.DataFrame(
            columns=[
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
        )

    def open_camera(
        self,
        index: int = CAM_INDEX,
        width: Optional[int] = FRAME_WIDTH,
        height: Optional[int] = FRAME_HEIGHT,
    ) -> cv2.VideoCapture:
        """
        Open a camera by its index and set frame dimensions.

        Args:
            index: Camera index (default from config)
            width: Frame width in pixels
            height: Frame height in pixels

        Returns:
            cv2.VideoCapture: Opened camera capture object

        Raises:
            RuntimeError: If camera cannot be opened
        """
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera with index {index}")

        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        return cap

    def preview_camera_loop(
        self,
        cap: cv2.VideoCapture,
        window_title: str = WINDOW_TITLE,
        exit_key: str = "q",
    ) -> None:
        """
        Show live camera preview with object detection until exit key is pressed.

        Args:
            cap: Camera capture object
            window_title: Window title for display
            exit_key: Key to press for exit
        """
        print("Camera opened successfully.")
        print(f"Press the '{exit_key}' key to close the window.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Could not read a frame from the camera.")
                break

            self._object_detection_process(frame)
            cv2.imshow(window_title, frame)

            if cv2.waitKey(1) & 0xFF == ord(exit_key):
                break

    def release_camera(cap: cv2.VideoCapture) -> None:
        """
        Release camera resources and destroy OpenCV windows.

        Args:
            cap: Camera capture object to release
        """
        cap.release()
        cv2.destroyAllWindows()

    def read_image_from_file(self, file_path: str) -> np.ndarray:
        """
        Read an image from file path.

        Args:
            file_path: Path to image file

        Returns:
            np.ndarray: Image as NumPy array

        Raises:
            FileNotFoundError: If image file cannot be read
        """
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image from path: {file_path}")
        return image

    def preview_image(
        self,
        image: np.ndarray,
        window_title: str = "Image Preview",
        exit_key: str = "q",
    ) -> None:
        """
        Preview a single image with object detection in a window.

        Args:
            image: Image array to display
            window_title: Window title for display
            exit_key: Key to press for exit
        """
        resized_image = cv2.resize(image, (FRAME_WIDTH, FRAME_HEIGHT))

        self._object_detection_process(resized_image)

        cv2.imshow(window_title, resized_image)
        print(f"Press '{exit_key}' key to close the image preview.")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def read_video_from_file(self, file_path: str) -> cv2.VideoCapture:
        """
        Open video file for reading.

        Args:
            file_path: Path to video file

        Returns:
            cv2.VideoCapture: Video capture object

        Raises:
            FileNotFoundError: If video file cannot be opened
        """
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not read video from path: {file_path}")
        return cap

    def preview_video(
        self,
        cap: cv2.VideoCapture,
        window_title: str = "Video Preview",
        exit_key: str = "q",
    ) -> None:
        """
        Preview video with object detection from VideoCapture object.

        Args:
            cap: Video capture object
            window_title: Window title for display
            exit_key: Key to press for exit
        """
        print("Video opened successfully.")
        print(f"Press the '{exit_key}' key to close the window.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("End of video or could not read a frame.")
                break

            self._object_detection_process(frame)
            cv2.imshow(window_title, frame)

            if cv2.waitKey(30) & 0xFF == ord(exit_key):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _object_detection_process(self, frame: np.ndarray) -> None:
        """
        Process frame with YOLO detection and draw results.

        Args:
            frame: Input frame to process (modified in-place)
        """
        detection_result = self._yolo_model.run_inference_on_frame(frame)

        if detection_result is not None:
            class_name, confidence, bbox = detection_result
            self._draw_detection_box(frame, bbox, class_name, confidence)

    @staticmethod
    def _draw_detection_box(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        class_name: str,
        confidence: float,
    ) -> None:
        """
        Draw detection bounding box and label on frame.

        Args:
            frame: Frame to draw on (modified in-place)
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            class_name: Detected class name
            confidence: Detection confidence score
        """
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 3)

        # Prepare text label
        label = f"{class_name} {confidence:.2f}"

        # Calculate text size for better positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Draw text background rectangle
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            BOX_COLOR,
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            FONT_COLOR,
            thickness,
        )

    # Public Entry Points
    def run_camera_preview(self) -> None:
        """Entry point for camera preview with object detection."""
        cap = None
        try:
            cap = self.open_camera()
            self.preview_camera_loop(cap, WINDOW_TITLE)
        except Exception as e:
            print(f"Error during camera preview: {e}")
        finally:
            if cap is not None:
                self.release_camera(cap)

    def run_image_preview(self) -> None:
        """Entry point for image preview with object detection."""
        try:
            image = self.read_image_from_file(IMG_INPUT_PATH)
            self.preview_image(image, "Image Preview")
        except Exception as e:
            print(f"Error during image preview: {e}")

    def run_video_preview(self) -> None:
        """Entry point for video preview with object detection."""
        cap = None
        try:
            cap = self.read_video_from_file(VIDEO_INPUT_PATH)
            self.preview_video(cap, "Video Preview")
        except Exception as e:
            print(f"Error during video preview: {e}")
        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()
