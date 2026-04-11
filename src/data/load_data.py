import pandas as pd

from consts import SRC_DIR, APPCAT_VARS

def load(file_path: str | None = None) -> pd.DataFrame:
    """ Load the dataset from a CSV file. """
    if file_path is None:
        file_path = SRC_DIR / "data" / "dataset_mood_smartphone.csv"
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is an index
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format

    for var in APPCAT_VARS + ['screen']:
        df.loc[df['variable'] == var, 'duration'] = pd.to_timedelta(df.loc[df['variable'] == var, 'value'], unit='s')
        df.loc[df['variable'] == var, 'end_time'] = df.loc[df['variable'] == var, 'time'] + df.loc[df['variable'] == var, 'duration']

    return df
