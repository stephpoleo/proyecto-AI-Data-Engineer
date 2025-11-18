SELECT
    source_id,
    source_type,
    time_window_10s,
    COUNT(*)        AS count_objects
FROM yolo_objects
GROUP BY source_id, source_type, time_window_10s