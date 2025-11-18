from src.etl.etl import ETL


def run_batch_etl_system():
    """
    Entry point to run the batch ETL system.
    """
    etl = ETL(output_path="data/output/")
    data = etl.extract()
    transformed_data = etl.transform(data)
    etl.load(transformed_data, clear_first=False)
