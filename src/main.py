from pathlib import Path
import pandas as pd

from data_analyser import load_data, Visualiser, Analyser


def main():
    # Load the data
    df = load_data()  # Pass None to use
    visualiser = Visualiser(df)
    analyser = Analyser(df)

    # Print describives of each variable
    visualiser.descriptives()

    # builtin and entertainment have negative minimum values, set any values smaller than 0 to 0.
    analyser.cap_variables(vars = ['appCat.builtin', 'appCat.entertainment'], cap=0.0)

    # import edited data back into Visualiser, check descriptives again
    visualiser.import_data(analyser.data)
    visualiser.descriptives() # minimum values for these variables are now 0.

    # Check which individuals have how many NAs for which variables
    visualiser.na_heatmap(save=True)

    # Check which individuals have how many NAs for which days
    visualiser.na_heatmap(date=True, save=True)

    # 15 individuals have NA values for the variables arousal and valence. No other NAs are present in the data. No NAs within timestamps of individuals either.





    # visualiser.individual_outlier_plot(save=True)
    # visualiser.timestamp_barcode(
    #     vars = ["mood", "activity", "screen", "circumplex.arousal"],
    #     save = True)
    # visualiser.timestamp_heatmap(save=True)

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
