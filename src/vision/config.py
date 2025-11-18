IMG_INPUT_PATH = "data/input/images/"
VIDEO_INPUT_PATH = "data/input/videos/"
OUTPUT_DATA_PATH = "data/output/"

CAM_INDEX = 0  # Change index to try different cameras. 0 is usually the default camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# YOLO
YOLO_MODEL_PATH = "models/yolov8n.pt"

ALLOWED_CLASSES = {
    "person",
    "car",
    "backpack",
    "bottle",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "book",
    "toothbrush",
    "suitcase",
    "chair",
    "cat",
    "dog",
    "bench",
    "toilet",
    "cup",
    "apple",
    "clock",
    "banana",
    "broccoli",
    "spoon",
    "microwave",
    "bottle",
    "wine glass",
    "dining table",
}

# Default colors
BOX_COLOR = (255, 0, 0)  # Blue (default)
FONT_COLOR = (255, 255, 255)  # White

# Color mapping for different object classes (BGR format)
CLASS_COLORS = {
    "person": (0, 255, 0),  # Green
    "car": (255, 0, 0),  # Blue
    "backpack": (0, 165, 255),  # Orange
    "bottle": (255, 255, 0),  # Cyan
    "tv": (128, 0, 128),  # Purple
    "laptop": (0, 255, 255),  # Yellow
    "mouse": (255, 192, 203),  # Pink
    "remote": (165, 42, 42),  # Brown
    "keyboard": (128, 128, 128),  # Gray
    "cell phone": (255, 20, 147),  # Deep Pink
    "book": (0, 128, 0),  # Dark Green
    "toothbrush": (255, 165, 0),  # Orange
    "suitcase": (30, 144, 255),  # Dodger Blue
}

# Default color for unknown classes
DEFAULT_COLOR = (255, 0, 255)  # Magenta
