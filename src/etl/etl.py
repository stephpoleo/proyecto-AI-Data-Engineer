import pandas as pd
from pathlib import Path
from .warehouse import (
    init_hive_schema,
    insert_into_hive,
    run_hive_analytics,
    clear_yolo_table,
)


class ETL:
    def __init__(self, output_path: str) -> None:
        self.output_path = output_path

    def extract(self):
        dfs = []
        for file_path in Path(self.output_path).iterdir():
            if not file_path.is_file():
                continue

            df = pd.read_csv(file_path)
            print(f"Dataframe cargado con {df.shape[0]} filas")
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Dataframe combinado con {combined_df.shape[0]} filas totales")
        return combined_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_row_count = df.shape[0]
        
        if self.has_nulls(df):
            df = self.remove_nulls(df)
        
        df = self.remove_invalid_coordinates(df)
        df = self.remove_out_of_range_confidence(df)
        df = self.filter_high_confidence(df, threshold=0.5)
        df = self.detect_and_remove_duplicates(df)

        self._print_transformation_summary(initial_row_count, df.shape[0])

        df = self.normalize_data(df)
        df = self.cast_data_types(df)
        df = self.create_feature_engineering_columns(df)
        return df

    def load(self, df: pd.DataFrame, clear_first: bool = False) -> None:
        init_hive_schema()
        if clear_first:
            clear_yolo_table(debug=True)
        print("\nCargando datos transformados en Hive...")
        insert_into_hive(df, debug=False)
        run_hive_analytics(debug=True, print_results=True)

    @staticmethod
    def has_nulls(df: pd.DataFrame) -> bool:
        has_nulls = df.isnull().values.any()
        if has_nulls:
            total_nulls = df.isnull().sum().sum()
            total_cells = df.shape[0] * df.shape[1]
            null_percentage = (total_nulls / total_cells) * 100
            print(f"Dataset tiene {total_nulls} valores nulos de {total_cells} celdas totales ({null_percentage:.2f}%)")
        return has_nulls

    @staticmethod
    def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
        print("Dataframe contiene valores nulos. Analizando y limpiando...")
        critical_cols = ['detection_id', 'x_min', 'y_min', 'x_max', 'y_max']
        df_cleaned = df.dropna(subset=critical_cols)
        return df_cleaned

    @staticmethod
    def remove_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
        print("Filtrando filas con coordenadas inválidas...")
        initial_count = df.shape[0]
        df_cleaned = df[~df.apply(ETL.clean_invalid_coordinates, axis=1)]
        removed_count = initial_count - df_cleaned.shape[0]
        print(
            f"Eliminadas {removed_count} filas con coordenadas inválidas. Filas restantes: {df_cleaned.shape[0]}"
        )
        return df_cleaned

    @staticmethod
    def clean_invalid_coordinates(row) -> bool:
        x_min, y_min, x_max, y_max = (
            row["x_min"],
            row["y_min"],
            row["x_max"],
            row["y_max"],
        )
        return x_min >= x_max or y_min >= y_max

    @staticmethod
    def remove_out_of_range_confidence(df: pd.DataFrame) -> pd.DataFrame:
        print("Filtrando valores de confianza fuera de rango...")
        initial_count = df.shape[0]
        df_cleaned = df[~df.apply(ETL.clean_out_of_range_confidence, axis=1)]
        removed_count = initial_count - df_cleaned.shape[0]
        print(
            f"Eliminadas {removed_count} filas con confianza fuera de rango. Filas restantes: {df_cleaned.shape[0]}"
        )
        return df_cleaned

    @staticmethod
    def clean_out_of_range_confidence(row) -> bool:
        confidence = row["confidence"]
        return confidence < 0.0 or confidence > 1.0

    @staticmethod
    def filter_high_confidence(
        df: pd.DataFrame, threshold: float = 0.5
    ) -> pd.DataFrame:
        print(f"Filtrando detecciones con confianza >= {threshold}...")
        initial_count = df.shape[0]
        df_filtered = df[
            df.apply(lambda row: ETL.keep_high_confidence(row, threshold), axis=1)
        ]
        removed_count = initial_count - df_filtered.shape[0]
        print(
            f"Eliminadas {removed_count} filas con baja confianza. Filas restantes: {df_filtered.shape[0]}"
        )
        return df_filtered

    @staticmethod
    def keep_high_confidence(row, threshold: float = 0.5) -> bool:
        confidence = row["confidence"]
        return confidence >= threshold

    @staticmethod
    def _print_transformation_summary(initial_count: int, final_count: int) -> None:
        cleaned_count = initial_count - final_count
        print("\n=== RESUMEN DE TRANSFORMACIÓN ===")
        print(f"Filas iniciales: {initial_count}")
        print(f"Filas finales: {final_count}")
        print(f"Filas limpiadas: {cleaned_count}")
        print("=" * 35)

    @staticmethod
    def detect_and_remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        print("Eliminando filas duplicadas...")
        initial_count = df.shape[0]
        key_cols = [
            "source_type",
            "source_id",
            "frame_number",
            "class_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]
        df_cleaned = df.drop_duplicates(subset=key_cols)
        removed_count = initial_count - df_cleaned.shape[0]
        print(
            f"Eliminadas {removed_count} filas duplicadas. Filas restantes: {df_cleaned.shape[0]}"
        )
        return df_cleaned

    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna({"class_name": "unknown"})
        df["detection_id"] = df["detection_id"].str.lower().replace(" ", "_")
        df["bbox_area_ratio"] = df["bbox_area_ratio"].round(3)
        df["center_x_norm"] = df["center_x_norm"].round(3)
        df["center_y_norm"] = df["center_y_norm"].round(3)
        df["timestamp_sec"] = df["timestamp_sec"].round(3)
        return df

    @staticmethod
    def cast_data_types(df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = [
            "source_type",
            "class_name",
            "position_region",
            "dominant_color_name",
        ]
        df[cat_cols] = df[cat_cols].astype("category")

        int_cols = [
            "frame_number",
            "class_id",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "width",
            "height",
            "frame_width",
            "frame_height",
            "center_x",
            "center_y",
            "area_pixels",
            "dom_r",
            "dom_g",
            "dom_b",
        ]
        df[int_cols] = df[int_cols].astype("int32")

        float_cols = [
            "confidence",
            "bbox_area_ratio",
            "center_x_norm",
            "center_y_norm",
            "timestamp_sec",
        ]
        df[float_cols] = df[float_cols].astype("float32")

        df["ingestion_date"] = pd.to_datetime(df["ingestion_date"])

        return df

    @staticmethod
    def create_feature_engineering_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Create additional feature engineering columns."""
        df["is_large_object"] = (df["bbox_area_ratio"] > 0.3).astype("int8")
        df["is_high_conf"] = (df["confidence"] >= 0.7).astype("int8")

        df["time_window_10s"] = df.apply(
            lambda row: (
                (row["timestamp_sec"] // 10) if row["source_type"] != "image" else 0
            ),
            axis=1,
        ).astype("int32")

        return df
