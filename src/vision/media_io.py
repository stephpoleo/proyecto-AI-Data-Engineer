"""
Media I/O handler for computer vision operations with YOLO detection.
Handles camera, image, and video processing with automatic data logging.
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import (
    CAM_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMG_INPUT_PATH,
    VIDEO_INPUT_PATH,
)
from .model import YoloModel
from .utils import (
    add_detection_to_dataframe,
    create_detection_dataframe_schema,
    draw_multiple_detections,
    extract_filename_from_path,
)


class MediaIO:
    """Media I/O handler for computer vision operations with YOLO detection."""

    def __init__(self) -> None:
        self._yolo_model = YoloModel()
        self.dataframe = pd.DataFrame(columns=create_detection_dataframe_schema())
        self._frame_counter = 0
        self._start_time = None

    def _reset_counters(self) -> None:
        """Reset frame counter and start time."""
        self._frame_counter = 0
        self._start_time = time.time()

    def get_df_detections(self) -> pd.DataFrame:
        """Get the current detections dataframe."""
        return self.dataframe

    def _process_detection(
        self, frame: np.ndarray, source_type: str, source_id: str, preview: bool = True
    ) -> None:
        """Process frame with YOLO detection and update dataframe."""
        has_detections = self._yolo_model.run_inference_on_frame(frame)

        if not has_detections:
            return

        detections = self._yolo_model.get_detections()

        for class_name, confidence, bbox, class_id in detections:
            self.dataframe = add_detection_to_dataframe(
                self.dataframe,
                source_type,
                source_id,
                frame,
                class_name,
                confidence,
                bbox,
                class_id,
                self._frame_counter,
                self._start_time,
            )

        if preview and detections:
            draw_multiple_detections(frame, detections)
            detection_summary = ", ".join(
                [f"{name}({conf:.2f})" for name, conf, _, _ in detections]
            )
            print(f"Frame {self._frame_counter}: {detection_summary}")

    # ==================== CAMERA OPERATIONS ====================

    @staticmethod
    def diagnose_cameras() -> list[int]:
        """Diagnose available cameras and return list of working indices."""
        working_cameras = []
        backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]

        print("Scanning for available cameras...")
        for index in range(0, 10):  # Check indices 0-9
            for backend in backends:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(
                            f"✓ Camera {index} working (backend: {backend}, resolution: {frame.shape[1]}x{frame.shape[0]})"
                        )
                        working_cameras.append(index)
                        cap.release()
                        break  # Found working camera at this index
                cap.release()

        if not working_cameras:
            print("✗ No working cameras found")
        else:
            print(f"Found {len(working_cameras)} working camera(s): {working_cameras}")

        return working_cameras

    def open_camera(
        self,
        index: int = CAM_INDEX,
        width: Optional[int] = FRAME_WIDTH,
        height: Optional[int] = FRAME_HEIGHT,
    ) -> cv2.VideoCapture:
        """Open camera with specified dimensions."""
        # Try different backends in order of preference
        backends = [cv2.CAP_V4L2, cv2.CAP_DSHOW, cv2.CAP_ANY]

        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                print(
                    f"Camera opened successfully with backend {backend} and index {index}"
                )
                if width is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height is not None:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
            cap.release()

        # If all backends fail, try different indices
        print(f"Failed to open camera with index {index}. Trying other indices...")
        for test_index in range(0, 4):  # Try indices 0-3
            for backend in backends:
                cap = cv2.VideoCapture(test_index, backend)
                if cap.isOpened():
                    print(f"Camera found at index {test_index} with backend {backend}")
                    if width is not None:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    if height is not None:
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    return cap
                cap.release()

        raise RuntimeError(
            f"Could not open any camera. Please check if a camera is connected and accessible."
        )

    def preview_camera_loop(
        self,
        cap: cv2.VideoCapture,
        preview: bool = True,
        window_title: str = "Camera preview",
        exit_key: str = "q",
    ) -> None:
        """Live camera preview with object detection."""
        if preview:
            print(f"Camera opened. Press '{exit_key}' to exit.")
        else:
            print("Camera processing started. Press Ctrl+C to stop.")

        self._reset_counters()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame from camera.")
                break

            self._process_detection(frame, "camera", "live_camera", preview=preview)

            if preview:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(1) & 0xFF == ord(exit_key):
                    break
            else:
                self._frame_counter += 1
                if self._frame_counter % 30 == 0:
                    print(f"Processed {self._frame_counter} frames...")

            self._frame_counter += 1

    @staticmethod
    def release_camera(cap: cv2.VideoCapture) -> None:
        """Release camera and close windows."""
        cap.release()
        cv2.destroyAllWindows()

    # ==================== IMAGE OPERATIONS ====================

    def read_image_from_file(self, file_path: str) -> np.ndarray:
        """Read image from file path."""
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {file_path}")
        return image

    def preview_image(
        self,
        image: np.ndarray,
        window_title: str = "Image Preview",
        exit_key: str = "q",
    ) -> None:
        """Preview image with object detection."""
        cv2.imshow(window_title, image)
        print(f"Press '{exit_key}' to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ==================== VIDEO OPERATIONS ====================

    def read_video_from_file(self, file_path: str) -> cv2.VideoCapture:
        """Open video file for reading."""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {file_path}")
        return cap

    def preview_video(
        self,
        cap: cv2.VideoCapture,
        video_path: str,
        preview: bool = True,
        window_title: str = "Video Preview",
        exit_key: str = "q",
    ) -> None:
        """Preview video with object detection."""
        if preview:
            print(f"Video opened. Press '{exit_key}' to exit.")
        else:
            print("Video processing started...")

        self._reset_counters()
        source_id = extract_filename_from_path(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break

            self._process_detection(frame, "video", source_id, preview=preview)

            if preview:
                cv2.imshow(window_title, frame)
                if cv2.waitKey(30) & 0xFF == ord(exit_key):
                    break
            else:
                if self._frame_counter % 30 == 0:
                    print(f"Processed {self._frame_counter} frames...")

            self._frame_counter += 1

        cap.release()
        if preview:
            cv2.destroyAllWindows()

    # ==================== PUBLIC ENTRY POINTS ====================

    def run_camera_process(self, preview: bool = True) -> None:
        """Entry point for live camera processing with optional preview."""
        cap = None
        try:
            cap = self.open_camera()
            self.preview_camera_loop(cap, preview=preview)
        except RuntimeError as e:
            print(f"Camera error: {e}")
            print("\nRunning camera diagnosis...")
            working_cameras = self.diagnose_cameras()

            if working_cameras:
                print(f"\nTrying to use camera {working_cameras[0]}...")
                try:
                    cap = self.open_camera(index=working_cameras[0])
                    self.preview_camera_loop(cap, preview=preview)
                except Exception as e2:
                    print(f"Failed to use camera {working_cameras[0]}: {e2}")
            else:
                print("\nNo cameras available. Please:")
                print("1. Check if a camera is connected")
                print("2. Check camera permissions")
                print(
                    "3. If the code is running on WSL, switch to Windows or use a Linux environment"
                )
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            if cap is not None:
                self.release_camera(cap)

    def run_image_process(self, preview: bool = True) -> None:
        """Entry point for image preview with detection."""
        try:
            for image_path in Path(IMG_INPUT_PATH).iterdir():
                if not image_path.is_file():
                    continue
                image = self.read_image_from_file(str(image_path))
                source_id = extract_filename_from_path(str(image_path))

                self._reset_counters()
                self._process_detection(image, "image", source_id, preview=preview)

                if preview:
                    self.preview_image(image)
        except Exception as e:
            print(f"Image error: {e}")

    def run_video_process(self, preview: bool = True) -> None:
        """Entry point for video processing with optional preview."""
        cap = None
        try:
            for video_path in Path(VIDEO_INPUT_PATH).iterdir():
                if not video_path.is_file():
                    continue
                cap = self.read_video_from_file(str(video_path))
                self.preview_video(cap, str(video_path), preview=preview)
        except Exception as e:
            print(f"Video error: {e}")
        finally:
            if cap is not None:
                cap.release()
                if preview:
                    cv2.destroyAllWindows()
