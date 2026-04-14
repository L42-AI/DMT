import pandas as pd

from consts import SRC_DIR

def load(file_path: str | None = None) -> pd.DataFrame:
    if file_path is None:
        file_path = SRC_DIR / "data" / "dataset_mood_smartphone.csv"
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is an index
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    return df
