import pandas as pd
from pathlib import Path
from pyhive import hive

HIVE_CONN_ARGS = dict(
    host="localhost",
    port=10000,
    username="steph",
    database="yolo_db",
    auth="NONE",
)

SQL_QUERIES_DIR = Path(__file__).resolve().parent / "queries"
MAX_ROWS_PER_INSERT = 200


def get_hive_connection():
    return hive.Connection(**HIVE_CONN_ARGS)


def load_sql(filename: str) -> str:
    """Lee un archivo .sql desde src/etl/sql/."""
    path = SQL_QUERIES_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def init_hive_schema() -> None:
    """
    Inicializa el esquema de Hive ejecutando un archivo .sql con DDL.
    """
    schema_path = SQL_QUERIES_DIR / "hive_schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"No encontré el archivo de esquema: {schema_path}")

    sql_text = schema_path.read_text(encoding="utf-8")

    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]

    conn = get_hive_connection()
    cur = conn.cursor()

    for stmt in statements:
        if stmt.startswith("--"):
            continue
        print(f"[Hive] Ejecutando:\n{stmt}\n")
        cur.execute(stmt)

    cur.close()
    conn.close()


def sql_literal(v):
    if pd.isna(v):
        return "NULL"
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    s = str(v).replace("'", "''")
    return f"'{s}'"


def filter_already_existing_detections(
    conn, df: pd.DataFrame, debug: bool = False
) -> pd.DataFrame:
    """
    Devuelve solo las filas cuyo detection_id NO existe en yolo_objects.
    Usa la conexión abierta a Hive (conn).
    """
    if df.empty:
        return df

    ids = df["detection_id"].dropna().unique().tolist()
    existing_ids: set[str] = set()

    cur = conn.cursor()

    chunk_size = 500
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i : i + chunk_size]
        id_list = ", ".join(sql_literal(x) for x in chunk)
        query = f"""
            SELECT detection_id
            FROM yolo_objects
            WHERE detection_id IN ({id_list})
        """
        cur.execute(query)
        for row in cur.fetchall():
            existing_ids.add(row[0])

    cur.close()

    if debug:
        print(f"[Hive] detection_id ya existentes en Hive: {len(existing_ids)}")

    mask_new = ~df["detection_id"].isin(existing_ids)
    df_masked = df[mask_new].copy()

    if debug:
        print(f"[Hive] Nuevas detecciones a insertar: {len(df_masked)} de {len(df)}")

    return df_masked

def insert_into_hive(df: pd.DataFrame, debug: bool = False) -> None:
    conn = get_hive_connection()
    cur = conn.cursor()
    table_name = "yolo_objects"

    cols = (
        "detection_id, source_type, source_id, frame_number, "
        "class_id, class_name, confidence, "
        "x_min, y_min, x_max, y_max, "
        "width, height, area_pixels, "
        "frame_width, frame_height, bbox_area_ratio, "
        "center_x, center_y, center_x_norm, center_y_norm, "
        "position_region, dominant_color_name, dom_r, dom_g, dom_b, "
        "timestamp_sec, ingestion_date, "
        "is_large_object, is_high_conf, time_window_10s"
    )

    for window, chunk in df.groupby("time_window_10s"):
        print(f"[Hive] Ventana {window}: {len(chunk)} filas")

        # troceamos esta ventana en pedazos manejables
        for start in range(0, len(chunk), MAX_ROWS_PER_INSERT):
            sub = chunk.iloc[start : start + MAX_ROWS_PER_INSERT]
            print(f"  -> Sub-batch {start}-{start+len(sub)-1} ({len(sub)} filas)")

            values_sql = []
            for _, row in sub.iterrows():
                tup = (
                    row["detection_id"],
                    row["source_type"],
                    row["source_id"],
                    int(row["frame_number"]),
                    int(row["class_id"]),
                    row["class_name"],
                    float(row["confidence"]),
                    int(row["x_min"]),
                    int(row["y_min"]),
                    int(row["x_max"]),
                    int(row["y_max"]),
                    int(row["width"]),
                    int(row["height"]),
                    int(row["area_pixels"]),
                    int(row["frame_width"]),
                    int(row["frame_height"]),
                    float(row["bbox_area_ratio"]),
                    int(row["center_x"]),
                    int(row["center_y"]),
                    float(row["center_x_norm"]),
                    float(row["center_y_norm"]),
                    row["position_region"],
                    row["dominant_color_name"],
                    int(row["dom_r"]),
                    int(row["dom_g"]),
                    int(row["dom_b"]),
                    float(row["timestamp_sec"]),
                    row["ingestion_date"],
                    int(row["is_large_object"]),
                    int(row["is_high_conf"]),
                    int(row["time_window_10s"]),
                )

                literals = [sql_literal(v) for v in tup]
                values_sql.append("(" + ", ".join(literals) + ")")

            query = f"INSERT INTO {table_name} ({cols}) VALUES " + ", ".join(values_sql)
            if debug:
                print(query[:500] + (" ... (truncado)" if len(query) > 500 else ""))

            cur.execute(query)

    cur.close()
    conn.close()



def run_hive_analytics(debug: bool = False, print_results: bool = True) -> dict:
    """
    Ejecuta las consultas analíticas en Hive.

    Devuelve un dict {nombre_query: DataFrame} y opcionalmente imprime
    los resultados en consola.
    """
    conn = get_hive_connection()
    cur = conn.cursor()

    queries = {
        "Objects per class": load_sql("objects_per_class.sql"),
        "People per video": load_sql("people_per_video.sql"),
        "Mean area per class": load_sql("bounding_box_mean_area_per_class.sql"),
        "Colors per class": load_sql("dominant_color_distrib.sql"),
        "Objects per time window": load_sql("objects_per_time_window.sql"),
    }
    resultados = {}

    for name, sql in queries.items():
        if debug:
            print(f"\n[Hive] Ejecutando query {name}:\n{sql}")

        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]
        df = pd.DataFrame(rows, columns=cols)
        resultados[name] = df

        if print_results:
            print("\n" + "=" * 80)
            print(f"RESULTADOS: {name}")
            print("=" * 80)
            print(df)

    cur.close()
    conn.close()
    return resultados


def clear_yolo_table(debug: bool = False) -> None:
    """Borra todas las filas de yolo_objects pero deja la tabla viva."""
    conn = get_hive_connection()
    cur = conn.cursor()

    sql = load_sql("clear_table.sql")
    if debug:
        print(f"[Hive] Ejecutando: {sql}")

    cur.execute(sql)

    cur.close()
    conn.close()
