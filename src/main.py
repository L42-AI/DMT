from data_analyser import Visualiser, Analyser
import data as _data

def main():
    # Load the data
    df = _data.load()
    visualiser = Visualiser(df)
    analyser = Analyser(df)
    
    analyser.process_outliers()

    # builtin and entertainment have negative minimum values, cap to zero.
    analyser.cap_variables(vars = ['appCat.builtin', 'appCat.entertainment'], cap=0.0)

    # Add Na durations for arousal and valence (might be useful later on)
    analyser.na_distribution_variable(variables = ['circumplex.valence', 'circumplex.arousal'])

    # Transform data to daily format for further EDA and cleaning
    analyser.daily_format(save = True)

    # === Daily Data Analysis ===

    visualiser.import_data(analyser)
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
    visualiser.show_correlations(save=True)



main()
