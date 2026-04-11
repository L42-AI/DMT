import sys

from visualiser import Visualiser
from analyser import Analyser
import data as _data

from features import extract_behavioural_features, extract_time_features

def main():
    # Load the data
    df = _data.load()
    visualiser = Visualiser(df)
    analyser = Analyser(df)
    
    analyser.process_outliers()

    # builtin and entertainment have negative minimum values, cap to zero.
    analyser.cap_variables(vars = ['appCat.builtin', 'appCat.entertainment'], cap=0.0)

    analyser.compute_gap_duration_for_variables(variables = ['circumplex.valence', 'circumplex.arousal'])

    # Transform data to daily format for further EDA and cleaning
    analyser.aggregate_daily(show = True)


    print(analyser.data.head())
    analyser.data = extract_time_features(analyser.data)
    analyser.data = extract_behavioural_features(analyser.data)
    analyser.data.to_csv('data/feature_engineered_data.csv', index=False)

    sys.exit(0)

    # === Daily Data Analysis ===

    # Show correlations between all variables
    visualiser.load(analyser.data, analyser.daily_data)

    visualiser.individual_outlier_plot(save=True)
    visualiser.var_correlations_per_id(save=True)
    visualiser.heatmap_missing_values_per_id(save=True)
    visualiser.val_dist_per_var(save=True)
    visualiser.var_dist_per_id(save=True)
    visualiser.ts_dist_per_id(save=True)
    visualiser.ts_dist_per_var(save=True)

    
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

    # Correlations between variables after imputation
    visualiser.var_correlations_per_id(save=True)



main()
