WITH colors_per_class AS (
    SELECT
        class_name,
        dominant_color_name,
        COUNT(*) AS count_objects_per_color
    FROM yolo_objects
    GROUP BY class_name, dominant_color_name
)
SELECT
    class_name,
    dominant_color_name,
    count_objects_per_color,
    ROUND(
        100.0 * count_objects_per_color
        / SUM(count_objects_per_color) OVER (PARTITION BY class_name),
        2
    ) AS pct_per_class_and_color
FROM colors_per_class
