from src.vision.media_io import MediaIO


def run_classification_system(mode: str):
    """
    Entry point to run the classification system.
    """
    media_io = MediaIO()
    if mode == "live_camera":
        media_io.run_camera_preview()
    elif mode == "image":
        media_io.run_image_preview()
    elif mode == "video":
        media_io.run_video_preview()
    else:
        raise ValueError(f"Unknown mode: {mode}")
