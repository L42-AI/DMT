from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from consts import APPCAT_VARS, USER_VARS, SENSOR_VARS

import helpers as _helpers

class Aggregator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def time_data(self, interval: int, unit: str, inplace: bool = False) -> pd.DataFrame:
        """
        Aggregates all sensor data into specified time intervals.

        Args:
            interval (int): The length of the interval (e.g., 1, 5, 10, 15, 30, 60).
            unit (str): The unit of time for the interval ('M' for minutes, 'H' for hours, 'D' for days).
        """

        def convert_to_seconds(value: float, unit: str) -> int:
            multiplier_map = {
                'M': 60,
                'H': 3600,
                'D': 86400
            }

            return int(value * multiplier_map[unit])

        assert interval in [1, 5, 10, 15, 30, 60], "Interval must be one of [1, 5, 10, 15, 30, 60]"
        assert unit in ['M', 'H', 'D'], "Unit must be one of 'M', 'H', or 'D'"

        seconds = convert_to_seconds(interval, unit)

        if unit == 'M': unit = 'min'

        agg_data = self.data[self.data['variable'].isin(APPCAT_VARS + ['screen'])].copy()
        agg_data['time'] = pd.to_datetime(agg_data['time'])
        agg_data.set_index('time', inplace=True)
        agg_data = agg_data.groupby(['id', 'variable']).resample(f'{interval}{unit.lower()}')['value'].sum().reset_index() 
        agg_data['value'] = agg_data['value'].where(agg_data['value'] <= seconds, seconds)

        # TODO: Consider scaling to 0-1 range by dividing by total possible seconds
        agg_data['value'] = agg_data['value'].round(3)

        agg_data.sort_values(['variable', 'id', 'time'], inplace=True)

        if inplace:
            new_data = pd.concat(
                [
                    self.data[~self.data['variable'].isin(APPCAT_VARS + ['screen'])],
                    agg_data
                ]
            ).reset_index(drop=True).sort_values(['variable', 'id', 'time'])
            
            self.data.__dict__.update(new_data.__dict__)
            
            return self.data

    def targets(self, inplace: bool = False) -> pd.DataFrame:
        """
        Mood is predicted as a mean per day, so we want to aggregate mood data into daily format. This is a simple mean aggregation per day per ID.
        """

        agg_data = self.data[self.data['variable'] == 'mood'].copy()
        agg_data['time'] = pd.to_datetime(agg_data['time'])
        agg_data.set_index('time', inplace=True)
        agg_data = agg_data.groupby(['id', 'variable']).resample('D')['value'].mean().reset_index()
        agg_data['value'] = agg_data['value'].round(2)
        agg_data.sort_values(['variable', 'id', 'time'], inplace=True)

        if inplace:
            new_data = pd.concat(
                [
                    self.data[self.data['variable'] != 'mood'],
                    agg_data
                ]
            ).reset_index(drop=True).sort_values(['variable', 'id', 'time'])

            self.data.__dict__.update(new_data.__dict__)
            return self.data

        return agg_data.sort_values(['variable', 'id', 'time'])

class Analyser:
    # === Constructor ===
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # Create date column
        self.data['date'] = self.data['time'].dt.date
        self.daily_data = None

        # Variable types
        self.scored_vars = USER_VARS
        self.sensor_vars = [var for var in self.data['variable'].unique() if var not in self.scored_vars]
        
        self.aggregate = Aggregator(data)

    def _handle_impossible_outliers(self):
        """
        Handle impossible outliers, f.e. negative values for variables that should not have negative values.
        Rules:
        - mood is 1-10
        - arousal and valence are -2 to 2
        - activity is 0-1
        - call made is 1
        - sms sent is 1
        - screen is time
        - all AppCat variables are time variables

        - All time variables follow time variable rules, f.e. no negative values        
        """

        print("Handling impossible outliers based on variable-specific rules...")
        print("Initial outlier counts:")
        print(self.data['value'].isna().sum())
        for var in self.data['variable'].unique():
            if var == 'mood':
                self.data.loc[(self.data['variable'] == var) & ((self.data['value'] < 1) | (self.data['value'] > 10)), 'value'] = np.nan
            elif var in ['circumplex.arousal', 'circumplex.valence']:
                self.data.loc[(self.data['variable'] == var) & ((self.data['value'] < -2) | (self.data['value'] > 2)), 'value'] = np.nan
            elif var == 'activity':
                self.data.loc[(self.data['variable'] == var) & ((self.data['value'] < 0) | (self.data['value'] > 1)), 'value'] = np.nan
            elif var == 'call':
                self.data.loc[(self.data['variable'] == var) & (self.data['value'] != 1), 'value'] = np.nan
            elif var == 'sms':
                self.data.loc[(self.data['variable'] == var) & (self.data['value'] != 1), 'value'] = np.nan
            elif var == 'screen' or var in APPCAT_VARS:
                self.data.loc[(self.data['variable'] == var) & (self.data['value'] < 0), 'value'] = np.nan
        
        print("Outlier counts after handling impossible outliers:")
        print(self.data['value'].isna().sum())

    def _handle_unlikely_outliers(self):
        """
        For each variable per ID, compute the mean and standard deviation

        Then order the z-scores of the values
        
        Calculate the distance between consecutive z-scores

        If gap is a statistical outlier (f.e. 3 IQRs above the third quartile), set value to NA. This means that we are looking for extreme outliers in the distribution of values per variable per ID, which is a sign of highly unlikely values.

        """

        print(self.data['value'].isna().sum())

        for var in self.data['variable'].unique():
            for id_val, group in self.data[self.data['variable'] == var].groupby('id'):
                values = group['value']
                z_scores = (values - values.mean()) / values.std()
                sorted_z = z_scores.sort_values()
                gaps = sorted_z.diff().abs()
                q1 = gaps.quantile(0.25)
                q3 = gaps.quantile(0.75)
                iqr = q3 - q1
                threshold = q3 + 3 * iqr
                outlier_indices = gaps[gaps > threshold].index
                self.data.loc[outlier_indices, 'value'] = np.nan

        print(self.data['value'].isna().sum())

    def process_outliers(self):
        """
        High-level method to deal with outliers. For this there are two groups:
        1. data points that are impossible 
        2. data points that are possible but highly unlikely (f.e. an individual with very extreme tendencies across many variables)
        """

        self._handle_impossible_outliers()

        # self._handle_unlikely_outliers()

    def apply_scaling(self):
        
        # Variable arousal and valence are -2 to 2 --> scale to -1 to 1 by dividing by 2
        div_by_2_mask = self.data['variable'].isin(['circumplex.arousal', 'circumplex.valence'])

        self.data.loc[div_by_2_mask, 'value'] = self.data.loc[div_by_2_mask, 'value'] / 2

        # TODO: Consider if good idea to scale target
        # Variable mood is 1-10 --> scale to 0-1 by applying (value - 1) / 9
        self.data.loc[self.data['variable'] == 'mood', 'value'] = (self.data.loc[self.data['variable'] == 'mood', 'value'] - 1) / 9

    # === Methods ===
    def compute_gap_duration_for_variables(self, variables: List[str]):
        """
        For given variables, compute duration of missing values. For consecutive NAs, the duration of missingness will be added. Observations without
        NAs have a duration of 0s by default.

        Args:
            variables (List[str]): List of variables with NAs. It is assumed that this specified list contains all variables in the dataset with missing values!
        """

        if 'gap_duration' not in self.data.columns:
            self.data['gap_duration'] = pd.Timedelta(0)

        for var in variables:
            # Select relevant variable data
            subset = self.data[self.data['variable'] == var].sort_values(['id', 'time'])
            
            # Look at one individual
            for _, person_data in subset.groupby('id'):
                # Create mask
                is_na = person_data['value'].isna()

                # Group observations in such a way that consecutive nas are in same group. We then have groups of nas and groups of observations
                group_id = (is_na != is_na.shift()).cumsum()

                # Iterate through every group
                for _, group in person_data.groupby(group_id):
                    # identify na group
                    if group['value'].isna().any():
                        # find valid measurements before and after na occurence(s)
                        prev_idx = group.index[0] - 1
                        next_idx = group.index[-1] + 1
                        
                        if prev_idx in person_data.index and next_idx in person_data.index:
                            # Compute na duration
                            duration = person_data.loc[next_idx, 'time'] - person_data.loc[prev_idx, 'time']

                            # Add duration to NA instances in subset
                            subset.loc[group.index, 'gap_duration'] = duration

            # Add duratiion to NA instances in original dataset, set durations for other variables 
            self.data.update(subset[['gap_duration']])

    def cap_variables(self, vars: List[str], cap: float, min: bool = True):
        """ Caps minimum and maximum values of variables. F.e. some variables should not have negative values, or some variables have outliers
            that need to be capped

        Args:
            vars (List[str]): List of variable names to cap the minimum of.
            cap (float)     : The cap value
            min (bool)      : Whether to cap minimum or maximum values. Default to minimum (True)
        """
        if min:
            mask = self.data['variable'].isin(vars) & (self.data['value'] < cap)
        else:
            mask = self.data['variable'].isin(vars) & (self.data['value'] > cap)
        self.data.loc[mask, 'value'] = cap # cap at 0? or set to NA?
            

    def design_mat(self):
        self.data['time'] = pd.to_datetime(self.data['time'])
        design_mat = self.data.pivot_table(index = ['id', 'time'],
                                           columns = 'variable',
                                           values = 'value',
                                           aggfunc = 'mean'
                                           )
        print(design_mat.head())


    def aggregate_daily(self, save: bool = False, show: bool = False):
        # In progress
        """ Aggregate all data into daily format. Aggregation method is mean for user-entered scores, and sum for 
        all other variables. Rows are instances, here defined as combination of (id, date). Columns are variables/
        attributes.

        Args:
            save (bool, optional): Whether to save as csv. Defaults to False.
            show (bool, optional): Whether to print a header (10 rows). Defaults to False.
        """
        
        # want a table of aggregate values for every combination of id, date, and variables
        sum_mask = self.data['variable'].isin(self.sensor_vars)
        mean_mask = self.data['variable'].isin(self.scored_vars)
        daily_data_sum = self.data[sum_mask].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
        daily_data_mean = self.data[mean_mask].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
        daily_data = pd.concat([daily_data_mean, daily_data_sum], axis = 1)
        
        # Create multi-index of ids and full range of dates 
        unique_ids = self.data['id'].unique()
        multi_index = pd.MultiIndex.from_product([unique_ids, _helpers.total_time_range(self.data['time'])], names=['id', 'date'])

        # Reindex the data
        daily_data = daily_data.reindex(multi_index).reset_index()

        daily_data = daily_data.melt(
            id_vars = ['id', 'date'],
            var_name = 'variable',
            value_name = 'value' 
        )


        if save:
            dir = Path('results/data')
            dir.mkdir(exist_ok=True, parents=True)
            daily_data.to_csv(dir / "daily_format.csv")

        if show:
            print(daily_data.head(10))
        
        self.daily_data = daily_data

    def impute(self, listwise_deletion: bool = True):
        # NOTE: In progress, imputation for scored variables still missing
        if self.daily_data is None:
            raise ValueError("Daily data is not defined")

        # sensor-data imputation
        sensor_mask = (self.daily_data['variable'].isin(self.sensor_vars)) & (self.daily_data['value'].isna())
        self.daily_data.loc[sensor_mask, 'value'] = 0

        # remove na-rows in scored variable data
        if listwise_deletion:
            scored_na_mask = (self.daily_data['variable'].isin(self.scored_vars)) & (self.daily_data['value'].isna())
            self.daily_data = self.daily_data[~scored_na_mask].reset_index(drop=True)
            return
        
        # Create individual dataframes of format (seq_length, vars + 1)
        ind_data = self.daily_data.groupby('id')
        for id, data in ind_data:
            wide_data = _helpers.wide_format_daily(data).reset_index()
        
            # TODO: NEED to align data to first day of observation!

            wide_data['date'] = (wide_data['date'] - wide_data['date'].iloc[0]).dt.days.values
            wide_data = wide_data.drop(columns = "id")
            # Save for CATSI
            dir = Path('src/data/catsi')
            dir.mkdir(exist_ok=True, parents=True)
            wide_data.to_csv(dir / f"{id}.csv")


    def get_suggested_transformations(self):
        """
        Analyzes the distribution of each variable and outputs a dictionary
        mapping the variable to suggested feature transformations.
        """
        
        distributions = [
            ("Normal", stats.norm),
            ("Exponential", stats.expon),
            ("Log-normal", stats.lognorm),
            ("Gamma", stats.gamma)
        ]
        
        # A simple mapping of distribution shapes to standard machine learning transformations
        transformation_map = {
            "Normal": "None (or Standard Scaling)",
            "Exponential": "Log transformation (e.g., np.log1p) or Square Root",
            "Log-normal": "Log transformation (e.g., np.log)",
            "Gamma": "Box-Cox transformation or Log transformation"
        }
        
        suggested_transformations = {}
        grouped_by_var = self.data.groupby('variable')
        
        for var_val, var_group in grouped_by_var:
            clean_data = var_group['value'].dropna().to_numpy()
            if len(clean_data) == 0:
                continue
                
            best_dist = None
            best_p = 0.0
            
            for dist_name, dist_obj in distributions:
                try:
                    # Handle domain constraints for Log-normal and Gamma
                    if dist_name in ["Log-normal", "Gamma"]:
                        fit_data = clean_data[clean_data > 0]
                        if len(fit_data) == 0:
                            continue
                    else:
                        fit_data = clean_data
                        
                    # Fit and test
                    params = dist_obj.fit(fit_data)
                    _, ks_pvalue = stats.kstest(fit_data, dist_obj.cdf, args=params)
                    
                    # Track the best performing distribution
                    if ks_pvalue > best_p:
                        best_p = ks_pvalue
                        best_dist = dist_name
                        
                except Exception:
                    continue # Silently skip failing mathematical fits
            
            # If the best fit is statistically significant (p > 0.05), recommend its pair
            if best_dist and best_p > 0.05:
                suggested_transformations[var_val] = transformation_map[best_dist]
            else:
                # If nothing fits well, recommend a robust power transformer
                suggested_transformations[var_val] = "Yeo-Johnson or Quantile Transformation"
                
        return suggested_transformations
