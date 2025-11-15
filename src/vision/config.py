CAM_INDEX = 0  # Change index to try different cameras. 0 is usually the default camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
WINDOW_TITLE = "Camera preview"

# YOLO
YOLO_MODEL_PATH = "models/yolov8n.pt"

ALLOWED_CLASSES = {
    "person",
    "backpack",
    "bottle",
    "tv",  # OJO: en COCO es "tv", no "tvmonitor"
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "toothbrush",
}

BOX_COLOR = (255, 0, 0)
FONT_COLOR = (255, 255, 255)
