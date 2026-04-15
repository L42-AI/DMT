import sys

from visualiser import Visualiser
from analyser import Analyser
from train import train
import data as _data
from features import extract_behavioural_features, extract_time_features

def main():
    # Load the data
    data = _data.load()
    visualiser = Visualiser(data)
    analyser = Analyser(data)

    analyser.process_outliers()
    analyser.aggregate.time_data(interval=1, unit='D', inplace=True)
    analyser.aggregate.activity(interval=1, unit='D', inplace=True)
    analyser.aggregate.communication_events(interval=1, unit='D', inplace=True)
    analyser.aggregate.reported_data(inplace=True)

    analyser.data.to_csv('data/aggregated_data.csv', index=False)
    analyser.impute(delete=False, catsi=True, epochs=10)
    analyser.data.to_csv('data/aggregated_data_after_impute.csv', index=False)

    print(analyser.data.isna().any())    
    
    train(analyser)

    # sys.exit(0)

    # === Daily Data Analysis ===

    # Show correlations between all variables
    # visualiser.load(analyser.data, analyser.daily_data)

    # visualiser.individual_outlier_plot(save=True)
    # visualiser.var_correlations_per_id(save=True)
    # visualiser.heatmap_missing_values_per_id(save=True)
    # visualiser.val_dist_per_var(save=True)
    # visualiser.var_dist_per_id(save=True)
    # visualiser.ts_dist_per_id(save=True)
    # visualiser.ts_dist_per_var(save=True)
    # visualiser.timestamp_heatmap(save=True)


    
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
    # analyser.impute(listwise_deletion= False)

    # Correlations between variables after imputation
    # visualiser.var_correlations_per_id(save=True)



main()
