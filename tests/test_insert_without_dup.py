# tests/test_dedup_insert.py
import pandas as pd
from src.etl.warehouse import (
    get_hive_connection,
    clear_yolo_table,
    insert_into_hive,
)


def test_insert_no_duplica_detection_id():
    """Goal: test that inserting a DataFrame with duplicate detection_id does not create duplicates in Hive."""
    clear_yolo_table()

    base_row = {
        "detection_id": "dup_1",
        "source_type": "image",
        "source_id": "img.jpg",
        "frame_number": 0,
        "class_id": 1,
        "class_name": "person",
        "confidence": 0.9,
        "x_min": 0,
        "y_min": 0,
        "x_max": 10,
        "y_max": 10,
        "width": 10,
        "height": 10,
        "area_pixels": 100,
        "frame_width": 100,
        "frame_height": 100,
        "bbox_area_ratio": 0.01,
        "center_x": 5,
        "center_y": 5,
        "center_x_norm": 0.05,
        "center_y_norm": 0.05,
        "position_region": "middle-center",
        "dominant_color_name": "gray",
        "dom_r": 100,
        "dom_g": 100,
        "dom_b": 100,
        "timestamp_sec": 0.0,
        "ingestion_date": "2025-11-18 00:00:00",
        "is_large_object": 0,
        "is_high_conf": 1,
        "time_window_10s": 0,
    }

    df1 = pd.DataFrame([base_row])
    insert_into_hive(df1)

    df2 = pd.DataFrame([base_row])
    insert_into_hive(df2)

    conn = get_hive_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM yolo_objects WHERE detection_id = 'dup_1'")
    count_dup = cur.fetchone()[0]
    cur.close()
    conn.close()

    assert count_dup == 1
