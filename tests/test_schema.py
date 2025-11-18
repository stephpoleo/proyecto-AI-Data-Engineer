from src.etl.warehouse import get_hive_connection


def test_yolo_table_exists():
    """Goal: test that init_hive_schema works by creating the yolo_objects table"""
    conn = get_hive_connection()
    cur = conn.cursor()
    cur.execute("SHOW TABLES LIKE 'yolo_objects'")
    assert cur.fetchone() is not None
    cur.close()
    conn.close()
