SELECT
    class_name,
    AVG(area_pixels)     AS pixels_avg_area,
    AVG(bbox_area_ratio) AS bbox_avg_area
FROM yolo_objects
GROUP BY class_name
ORDER BY bbox_avg_area DESC