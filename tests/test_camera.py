import pytest

from src.vision.media_io import MediaIO


def test_diagnose_cameras_returns_list():
    """Goal: test that diagnose_cameras returns a list of camera IDs."""
    media_io = MediaIO()
    cameras = media_io.diagnose_cameras()

    assert isinstance(cameras, list)

    for cam_id in cameras:
        assert isinstance(cam_id, int)


def test_at_least_one_camera_if_available():
    """Goal: test that at least one camera is detected if available. If not, discard the test."""
    media_io = MediaIO()
    cameras = media_io.diagnose_cameras()

    if not cameras:
        pytest.skip(
            "No hay cámara disponible en este entorno (WSL, cambiar a Windows o Linux)"
        )

    assert len(cameras) > 0
