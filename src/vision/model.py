import math

from ultralytics import YOLO
from .config import YOLO_MODEL_PATH
from .config import ALLOWED_CLASSES


class YoloModel:
    def __init__(self) -> None:
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names
        self.detections = []

    def get_detections(self) -> list:
        """Get all detections from the last inference."""
        return self.detections

    def get_class_names(self) -> list[str]:
        """Get list of detected class names."""
        return [detection[0] for detection in self.detections]

    def get_coordinates(self) -> list[tuple[int, int, int, int]]:
        """Get list of bounding box coordinates."""
        return [detection[2] for detection in self.detections]

    def get_confidences(self) -> list[float]:
        """Get list of confidence scores."""
        return [detection[1] for detection in self.detections]

    def get_class_ids(self) -> list[int]:
        """Get list of class IDs."""
        return [detection[3] for detection in self.detections]

    def get_class_name(self) -> str | None:
        return self.detections[0][0] if self.detections else None

    def get_coordinates_single(self) -> tuple[int, int, int, int] | None:
        return self.detections[0][2] if self.detections else None

    def get_confidence(self) -> float | None:
        return self.detections[0][1] if self.detections else None

    def _get_class_id_from_name(self, class_name: str) -> int:
        """
        Get class ID from class name using YOLO model.

        Args:
            class_name: Name of the class

        Returns:
            int: Class ID number
        """
        for class_id, name in self.class_names.items():
            if name == class_name:
                return class_id
        return 0

    def run_inference_on_frame(self, frame):
        """
        Runs YOLO inference on a single BGR frame (OpenCV format)
        and stores all valid detections in self.detections list.
        """
        self.detections = []

        results = self.model(frame, stream=True)
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                coordinates = (x_min, y_min, x_max, y_max)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, str(cls))

                if class_name not in ALLOWED_CLASSES:
                    continue

                detection = (class_name, confidence, coordinates, cls)
                self.detections.append(detection)

        return len(self.detections) > 0
