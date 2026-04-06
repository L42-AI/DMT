from pathlib import Path
import pandas as pd

from data_analyser import load_data, Visualiser, Analyser


def main():
    # Load the data
    df = load_data()  # Pass None to use
    visualiser = Visualiser(df)
    analyser = Analyser(df)

    # visualiser.timestamp_barcode(
    #     vars = ["mood", "activity", "screen", "circumplex.arousal"],
    #     save = True)
    visualiser.timestamp_heatmap(save=True)
    # visualiser.descriptives()
    # visualiser.datapoint_counts_per_id()
    # visualiser.timestamp_distribution_per_id()
    # visualiser.timestamp_distribution_per_var()
    # visualiser.value_distribution_per_id()
    # visualiser.value_distribution_per_variable()
    # visualiser.visualize_value_distribution_per_variable()
    # analyser.daily_format()
    # pprint(analyser.get_suggested_transformations())
    

main()
