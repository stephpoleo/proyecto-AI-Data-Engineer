from src.etl.etl import ETL


def run_batch_etl_system():
    """
    Entry point to run the batch ETL system.
    """
    etl = ETL(output_path="data/output/")
    data = etl.extract()
    transformed_data = etl.transform(data)
    if transformed_data.empty:
        print("No hay datos para cargar después de la transformación.")
        return
    etl.load(transformed_data, clear_first=True)
