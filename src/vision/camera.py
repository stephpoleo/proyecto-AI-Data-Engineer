import cv2

from .config import CAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, WINDOW_TITLE


def open_camera(
    index: int = CAM_INDEX,
    width: int | None = FRAME_WIDTH,
    height: int | None = FRAME_HEIGHT,
) -> cv2.VideoCapture:
    """
    Open a camera by its index and set frame width and height if provided.
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera with index {index}")

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def preview_loop(
    cap: cv2.VideoCapture,
    window_title: str = WINDOW_TITLE,
    exit_key: str = "q",
) -> None:
    """
    Show the camera preview until the `exit_key` is pressed.
    """
    print("Camera opened successfully.")
    print(f"Press the '{exit_key}' key to close the window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read a frame from the camera.")
            break

        cv2.imshow(window_title, frame)

        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            break


def release_camera(cap: cv2.VideoCapture) -> None:
    """
    Release the camera and destroy OpenCV windows.
    """
    cap.release()
    cv2.destroyAllWindows()


def run_camera_preview() -> None:
    """
    Entry point to get camera preview.
    """
    try:
        cap = open_camera()
        preview_loop(cap, WINDOW_TITLE)
    finally:
        release_camera(cap)
