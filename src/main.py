from pathlib import Path
import pandas as pd

from data_analyser import load_data, Visualiser, Analyser


def main():
    # Load the data
    df = load_data()  # Pass None to use
    visualiser = Visualiser(df)
    analyser = Analyser(df)
    


    # builtin and entertainment have negative minimum values, cap to zero.
    analyser.cap_variables(vars = ['appCat.builtin', 'appCat.entertainment'], cap=0.0)

    # Add Na durations for arousal and valence (might be useful later on)
    analyser.na_distribution_variable(variables = ['circumplex.valence', 'circumplex.arousal'])

    # Transform data to daily format for further EDA and cleaning
    analyser.daily_format(save = True)

    # === Daily Data Analysis ===

    # Show correlations between all variables
    visualiser.import_data(analyser)
    visualiser.show_correlations(save=True)
    """
    Within-individual correlations are substantial, depending on the individual. This means that the imputation method should be cross-sectional as well as longitudinal.
    It also suggests that our model should probably be on an individual level and not on a group level...
    """

    # Impute missing daily data
    """ 
    For sensor data:
    We assume that the data is missing in the aggregated format because the applications associated to the sensors were not used.
    The strategy here is to impute 0 for all sensor data, since missing sensor data in aggregated daily format means that the 
    individual did not use that phone/app category for the entire day. This should be quite straightforward

    For scored data:
    I'm thinking of the CATSI imputation algorithm, which is a context-aware time-series imputation technique"""
    analyser.impute()

    # # Check which individuals have how many NAs for which variables
    # visualiser.na_heatmap(save=True)

    # # Check which individuals have how many NAs for which days
    # visualiser.na_heatmap(date=True, save=True)

    # # 15 individuals have NA values for the variables arousal and valence. No other NAs are present in the data. No NAs within timestamps of individuals either.

    # visualiser.pairplot()

    # The outlier tendency of individuals does not match the the outlier extremity observed in the distributions -> Not individual outliers, but specific extreme outliers across individuals
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
    # pprint(analyser.get_suggested_transformations())
    

main()
