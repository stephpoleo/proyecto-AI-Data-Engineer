from src.etl.warehouse import get_hive_connection

def test_hive_connection():
    """ Goal: test that we can connect to Hive and list databases, including 'yolo_db' """
    conn = get_hive_connection()
    cur = conn.cursor()

    cur.execute("SHOW DATABASES")
    rows = cur.fetchall()

    dbs = {name for (name,) in rows}

    assert "default" in dbs
    assert "yolo_db" in dbs

    cur.close()
    conn.close()