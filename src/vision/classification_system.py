from datetime import datetime
from src.vision.media_io import MediaIO
from .utils import save_dataframe_to_csv


def run_classification_system(mode: str):
    """
    Entry point to run the classification system.
    """
    media_io = MediaIO()
    if mode == "live_camera":
        media_io.run_camera_process()
    elif mode == "image":
        media_io.run_image_process(preview=True)
    elif mode == "video":
        media_io.run_video_process(preview=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df_with_detections = media_io.get_df_detections()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dataframe_to_csv(df_with_detections, f"detections_{timestamp}.csv")
