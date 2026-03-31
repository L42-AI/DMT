from pathlib import Path
import pandas as pd


def main():
    # Load the data
    data_path = Path("data/dataset_mood_smartphone.csv")
    df = pd.read_csv(data_path)
    grouped = df.groupby("id")
    for id, group in grouped:
        print(f"ID: {id}")
        group

main()