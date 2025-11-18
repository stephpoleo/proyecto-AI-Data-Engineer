SELECT
    class_id,
    class_name,
    COUNT(DISTINCT detection_id) AS total_objetos
FROM yolo_objects
GROUP BY class_id, class_name
