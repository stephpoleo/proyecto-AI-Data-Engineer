from src.vision.classification_system import run_classification_system


if __name__ == "__main__":
    program_mode = ["live_camera", "image", "video"]
    run_classification_system(program_mode[1])  # Change index to select mode
