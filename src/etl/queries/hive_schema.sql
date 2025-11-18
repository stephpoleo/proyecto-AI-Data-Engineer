CREATE DATABASE IF NOT EXISTS yolo_db
LOCATION 'hdfs:///cursobsg/database/yolo_db.db';

USE yolo_db;

DROP TABLE IF EXISTS yolo_objects;

CREATE EXTERNAL TABLE IF NOT EXISTS yolo_objects (
  detection_id         STRING,
  source_type          STRING,
  source_id            STRING,
  frame_number         INT,
  class_id             INT,
  class_name           STRING,
  confidence           DOUBLE,
  x_min                INT,
  y_min                INT,
  x_max                INT,
  y_max                INT,
  width                INT,
  height               INT,
  area_pixels          INT,
  frame_width          INT,
  frame_height         INT,
  bbox_area_ratio      DOUBLE,
  center_x             INT,
  center_y             INT,
  center_x_norm        DOUBLE,
  center_y_norm        DOUBLE,
  position_region      STRING,
  dominant_color_name  STRING,
  dom_r                INT,
  dom_g                INT,
  dom_b                INT,
  timestamp_sec        DOUBLE,
  ingestion_date       TIMESTAMP,
  is_large_object      TINYINT,
  is_high_conf         TINYINT,
  time_window_10s      INT
)
STORED AS PARQUET
LOCATION 'hdfs:///cursobsg/tables/yolo_objects';

