import pandas as pd
from src.etl.etl import ETL

def test_create_feature_engineering_columns_image():
    """Test feature engineering columns for image source type, which should set time_window_10s to 0."""
    df = pd.DataFrame(
        [
            {
                "bbox_area_ratio": 0.5,
                "confidence": 0.9,
                "timestamp_sec": 12.0,
                "source_type": "image",
            }
        ]
    )

    out = ETL.create_feature_engineering_columns(df)

    assert "is_large_object" in out.columns
    assert "is_high_conf" in out.columns
    assert "time_window_10s" in out.columns
    assert out.loc[0, "is_large_object"] == 1
    assert out.loc[0, "is_high_conf"] == 1
    assert out.loc[0, "time_window_10s"] == 0


def test_create_feature_engineering_columns_video():
    """Test feature engineering columns for video source type, which should compute time_window_10s based on timestamp_sec."""
    df = pd.DataFrame(
        [
            {
                "bbox_area_ratio": 0.5,
                "confidence": 0.9,
                "timestamp_sec": 12.0,
                "source_type": "video",
            }
        ]
    )

    out = ETL.create_feature_engineering_columns(df)

    assert "is_large_object" in out.columns
    assert "is_high_conf" in out.columns
    assert "time_window_10s" in out.columns
    assert out.loc[0, "is_large_object"] == 1
    assert out.loc[0, "is_high_conf"] == 1
    assert out.loc[0, "time_window_10s"] == 1