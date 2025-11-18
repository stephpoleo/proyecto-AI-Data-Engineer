SELECT
    source_id AS video_file_name,
    COUNT(*) AS people_count
FROM yolo_objects
WHERE class_name = 'person' and source_type = 'video'
GROUP BY source_id