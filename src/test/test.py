def test_hive_connection():
    conn = hive.Connection(
        host="localhost",
        port=10000,
        username="steph",
        database="default",
        auth="NONE",
    )
    cur = conn.cursor()
    cur.execute("SHOW DATABASES")
    print(cur.fetchall())
    cur.close()
    conn.close()