import pandas as pd
from pathlib import Path


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

    def transform(self):
        pass

    def load(self):
        pass
