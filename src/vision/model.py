import math

from ultralytics import YOLO
from .config import YOLO_MODEL_PATH
from .config import ALLOWED_CLASSES


class YoloModel:
    def __init__(self) -> None:
        self.model = YOLO(YOLO_MODEL_PATH)
        self.class_names = self.model.names

    def run_inference_on_frame(
        self, frame
    ) -> tuple[str, float, tuple[int, int, int, int]] | None:
        """
        Runs YOLO inference on a single BGR frame (OpenCV format)
        and returns detection data or None if no valid detection found.
        """
        results = self.model(frame, stream=True)
        for r in results:
            for box in r.boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                confidence = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = self.class_names.get(cls, str(cls))

                if class_name not in ALLOWED_CLASSES:
                    continue

                return class_name, confidence, (x_min, y_min, x_max, y_max)

        return None
