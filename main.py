from src.vision.classification_system import run_classification_system
from src.etl.batch_etl_system import run_batch_etl_system


if __name__ == "__main__":
    program_mode = ["live_camera", "image", "video"]
    run_classification_system(program_mode[2])  # Change index to select mode
    run_batch_etl_system()
