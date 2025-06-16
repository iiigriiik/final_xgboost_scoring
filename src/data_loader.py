import pandas as pd


def load_data(path):
    """Загрузка данных из .parquet файла"""
    print(f"[INFO] Загрузка данных из {path}")
    return pd.read_parquet(path)
