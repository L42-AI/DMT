from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

from imputegap.recovery.manager import TimeSeries
from imputegap.recovery.contamination import GenGap
from imputegap.tools import utils

from IPython.display import display

# Globals
USER_VARS = ['circumplex.arousal', 'circumplex.valence', 'mood']

def load_data(file_path: str | None = None) -> pd.DataFrame:
    """ Load the dataset from a CSV file. """
    if file_path is None:
        file_path = Path("src/data/dataset_mood_smartphone.csv")
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is an index
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    return df


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

    # === Helpers ===     
    def _total_time_range(self):
        """ Returns the total range of time observed in the whole dataset """
        min_datetime = self.data['time'].dt.date.min()
        max_datetime = self.data['time'].dt.date.max()
        range_datetime = pd.date_range(min_datetime, max_datetime)
        return range_datetime
    
    # === Methods ===
    def na_distribution_variable(self, variables: List[str]):
        """For given variables, compute duration of missing values. For consecutive NAs, the duration of missingness will be added. Observations without
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


    def daily_format(self, save: bool = False, show: bool = False):
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
        multi_index = pd.MultiIndex.from_product([unique_ids, self._total_time_range()], names=['id', 'date'])

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

    def impute(self):
        # NOTE: In progress, imputation for scored variables still missing
        if self.daily_data is None:
            raise ValueError("Daily data is not defined")

        # sensor-data imputation
        for row in self.daily_data.index:
            
            if self.daily_data.loc[row, 'variable'] in self.sensor_vars and pd.isna(self.daily_data.loc[row, 'value']):
                self.daily_data.loc[row, 'value'] = 0
        
        # scored-data imputation
        # individual and variable level
        for (id, var), data in self.daily_data.groupby(['id', 'variable']):
            if var in self.scored_vars:
                print(f"{id}, {var}")
                mask_na = data['value'].isna()
                print(mask_na)


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

    # NOTE: IN Progress
    def extract_dataset_outliers_fast(self, threshold=0.99, local_weight=0.75, variance_buffer=0.15):
        """
        Extracts outliers with a looser tolerance to reduce false positives.
        
        Args:
            threshold (float): Increased to 0.99 to only flag the most extreme 1% of data.
            local_weight (float): Weight of personal vs peer distribution.
            variance_buffer (float): Prevents the "micro-variance trap" by ensuring a user's 
                                     local standard deviation never drops below a percentage 
                                     (e.g., 15%) of the peer standard deviation.
        """
        df = self.data.dropna(subset=['value']).copy()
        
        df['val_sq'] = df['value'] ** 2
        
        # Local aggregations
        local_aggs = df.groupby(['id', 'variable']).agg(
            n_local=('value', 'count'),
            sum_local=('value', 'sum'),
            sumsq_local=('val_sq', 'sum')
        ).reset_index()
        
        # Global aggregations
        global_aggs = local_aggs.groupby('variable').agg(
            n_global=('n_local', 'sum'),
            sum_global=('sum_local', 'sum'),
            sumsq_global=('sumsq_local', 'sum')
        ).reset_index()
        
        df = df.merge(local_aggs, on=['id', 'variable'])
        df = df.merge(global_aggs, on='variable')
        
        # Peer aggregations (Global - Local)
        df['n_peer'] = df['n_global'] - df['n_local']
        df['sum_peer'] = df['sum_global'] - df['sum_local']
        df['sumsq_peer'] = df['sumsq_global'] - df['sumsq_local']
        
        def calc_std(n, sums, sumsq):
            variance = (sumsq - (sums**2 / np.maximum(n, 1))) / np.maximum(n - 1, 1)
            return np.sqrt(np.maximum(variance, 0))
            
        # Calculate raw standard deviations
        df['mean_local'] = np.where(df['n_local'] > 0, df['sum_local'] / df['n_local'], 0)
        raw_std_local = calc_std(df['n_local'], df['sum_local'], df['sumsq_local'])
        
        df['mean_peer'] = np.where(df['n_peer'] > 0, df['sum_peer'] / df['n_peer'], 0)
        df['std_peer'] = calc_std(df['n_peer'], df['sum_peer'], df['sumsq_peer'])
        
        # Apply the Variance Buffer to loosen strict local tracking
        # If std_peer is 0, fallback to 1e-9 to prevent true division by zero
        safe_std_peer = np.where(df['std_peer'] == 0, 1e-9, df['std_peer'])
        df['std_local'] = np.maximum(raw_std_local, safe_std_peer * variance_buffer)
        
        # Calculate t-statistics
        t_local = (df['value'] - df['mean_local']) / df['std_local']
        t_peer = (df['value'] - df['mean_peer']) / safe_std_peer
        
        # Calculate Probabilities
        p_val_local = stats.t.sf(np.abs(t_local), df=np.maximum(df['n_local'] - 1, 1)) * 2
        prob_local = np.where(df['n_local'] >= 2, 1.0 - p_val_local, 0.0)
        
        p_val_peer = stats.t.sf(np.abs(t_peer), df=np.maximum(df['n_peer'] - 1, 1)) * 2
        prob_peer = np.where(df['n_peer'] >= 2, 1.0 - p_val_peer, 0.0)
        
        peer_weight = 1.0 - local_weight
        conditions = [
            (df['n_local'] < 2) & (df['n_peer'] >= 2),
            (df['n_peer'] < 2) & (df['n_local'] >= 2),
            (df['n_local'] >= 2) & (df['n_peer'] >= 2)
        ]
        choices = [
            prob_peer,
            prob_local,
            (prob_local * local_weight) + (prob_peer * peer_weight)
        ]
        df['outlier_probability'] = np.select(conditions, choices, default=0.0)
        
        flagged_outliers = df[df['outlier_probability'] >= threshold].copy()
        
        cols_to_drop = [
            'val_sq', 'n_local', 'sum_local', 'sumsq_local', 
            'n_global', 'sum_global', 'sumsq_global',
            'n_peer', 'sum_peer', 'sumsq_peer', 
            'mean_local', 'std_local', 'mean_peer', 'std_peer'
        ]
        flagged_outliers = flagged_outliers.drop(columns=cols_to_drop)
        
        return flagged_outliers.sort_values(by=['variable', 'outlier_probability'], ascending=[True, False])
    
    
class Visualiser:
    def __init__(self, data: pd.DataFrame):
        self.data = data

        # Create date column
        self.data['date'] = self.data['time'].dt.date
        self.daily_data = None

    # === Helpers ===

    # === Methods ===
    def import_data(self, analyser: Analyser):
        """Import new data to Visualizer, f.e. after using Analyser class to edit data.

        Args:
            data (pd.DataFrame): The data to import. 
        """
        self.data = analyser.data
        self.daily_data = analyser.daily_data

    def descriptives(self):
        ''' Descriptive statistics for all variables of the dataset '''
        data_audit = self.data.groupby('variable')['value'].describe()
        print(data_audit, '\n')


    def individual_outlier_plot(self, save: bool = False):
        """Plot the central tendencies of individuals across all variables. To show whether indivual tendencies for certain
           variables are outside the norm for some variables. F.e. an individual who has very extreme tendencies across many
           variables might be problematic.

        Args:
            save (bool, optional): Toggle saving vs showing plots. Defaults to showing(False).
        """
        dir = Path("results/eda")
        stats = self.data.groupby(['id', 'variable'])['value'].agg(['mean', 'std']).reset_index()
        means = stats.pivot(index = 'id', columns='variable', values='mean')        
        z_scores = (means - means.mean()) / means.std()
        plt.figure(figsize=(12, 8))
        sns.heatmap(z_scores, annot=True, cmap = "RdBu_r", center=0)
        plt.title("Individual Outlier Profile (Z-scores of Means)")
        if save:
            dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(dir / "individual_outlier_plot.png")
            plt.close()
        else:
            plt.show()

    def na_heatmap(self, date: bool = False, save: bool = False):
        """Visualize the presence of NAs amongs individual/variable or individual/date combinations

        Args:
            date (bool, optional): Whether you want to visualize NAs in dates instead of variables. Defaults to False.
            save (bool, optional): Whether you want to save instead of plot. Defaults to False.
        """
        dir = Path('results/eda')

        col = 'date' if date else 'variable'
        target_col = 'time' if date else 'value'

        grouped = self.data.groupby(["id", col])[target_col]
        counts = grouped.size()
        na_counts = grouped.apply(lambda x: x.isna().sum())  
        na_summary = ((na_counts / counts) * 100).unstack(fill_value=0).astype(float)
        mask = na_summary == 0

        # plot
        ax = plt.gca()
        ax.set_facecolor('#f0f0f0') # A neutral light grey for "Perfect Data"
        plt.figure(figsize=(12, 12))
        sns.heatmap(na_summary,
                    mask=mask,
                    annot=True,
                    cmap="YlOrRd",
                    fmt=".2f",
                    cbar=True,
                    linewidths=.5,
                    linecolor="lightgrey",
                    cbar_kws={'label': 'Percentage Missing (%)'})
        plt.title("Percentage of Missing Values in Individuals")
        plt.xlabel("Variable")
        plt.ylabel("User")

        if save:
            dir.mkdir(parents=True, exist_ok=True)
            filepath = dir / f"na_heatmap_{col}.png"
            plt.savefig(filepath)
            plt.close()
        else:
            plt.show()

    def show_na_distribution(self):
        """ Show the distribution of NA durations for the original data. In hours """
        duration_in_hours = self.data.loc[self.data['gap_duration'] > pd.Timedelta(0),'gap_duration'].dt.total_seconds() / 3600
        plt.hist(duration_in_hours)
        plt.show()

    def datapoint_counts_per_id(self):
        """ Visualize the number of datapoints per id. """
        grouped = self.data.groupby('id').size()
        sorted_idx = grouped.sort_values().index
        grouped = grouped[sorted_idx]
        plt.bar(grouped.index, grouped.values)
        plt.xlabel('ID')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.ylabel('Number of Datapoints')
        plt.title('Number of Datapoints per ID')
        plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better readability
        plt.show()

    def timestamp_distribution_per_id(self):
        """ Visualize the distribution of timestamps per id. """
        self.data['time'] = pd.to_datetime(self.data['time'])
        grouped = self.data.groupby('id')
        for id_val, group in grouped:
            plt.figure(figsize=(10, 4))
            plt.hist(group['time'], bins=50, color='blue', alpha=0.7)
            plt.xlabel('time')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of times for ID {id_val}')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def timestamp_distribution_per_var(self):
        """ Visualize the distribution of timestamps per variable. """
        self.data['time'] = pd.to_datetime(self.data['time'])
        grouped = self.data.groupby('variable')
        for var_val, group in grouped:
            plt.figure(figsize=(10, 4))
            plt.hist(group['time'], bins=50, color='blue', alpha=0.7)
            plt.xlabel('time')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of times for variable {var_val}')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def timestamp_barcode(self, vars: List[str], save: bool = False):
        """Barcode plots to visualize the times of the variable occurences in the data for every individual


        Args:
            vars (List[str]): Variables to compare
            save (bool, optional): Toggle saving vs showing plots. Defaults to showing(False).
        """
        # Path for saving
        dir = Path('results/eda/barcode')

        grouped = self.data.groupby("id")
        for id, group in grouped:
            plt.figure(figsize = (15, 6))

            for i, var in enumerate(vars):    
                # series of timestamps of variable for user
                var_times = group[group['variable'] == var]['time']
                # plot vertical lines, use i for plotting vertically
                plt.vlines(var_times, i - 0.4, i + 0.4, color = 'midnightblue', alpha = 0.4, linewidth = 0.5)

            plt.yticks(range(len(vars)), vars)
            plt.title(f"ID: {id}, Observation Density")
            plt.xlabel("Time")

            if save:
                dir.mkdir(parents=True, exist_ok=True)
                save_name = dir / f"barcode_{id}.png"
                plt.savefig(save_name, bbox_inches = 'tight')
                plt.close()
                continue
            else:
                plt.show()
    
    def timestamp_heatmap(self, save: bool = False):
        """Plots heatmap of daily variable occurences for every user on every day. For each variable a
           plot is generated. Darkblue means that the user has at least one entry of the variable
           for that day, while white means the user has no entry.

        Args:
            save (bool, optional): Toggle saving vs showing plots. Defaults to showing (False).
        """

        # path for saving
        dir = Path("results/eda/multi_barcode")

        grouped = self.data.groupby('variable')
        for var, group in grouped:

            # reshape and aggregate variable data to desired format -> (id, date), with entries being binary indicators whether 
            # there are any variable values for that individual and day.
            presence = group.pivot_table(
                index = 'id',
                columns = 'date',
                values = 'value',
                aggfunc = 'count'
            ).fillna(0)
            presence_bin = (presence > 0).astype(int)

            # plot
            plt.figure(figsize=(16, 8))
            sns.heatmap(presence_bin,
                        cmap="YlGnBu",
                        cbar=False,
                        linewidths=.5,
                        linecolor="lightgrey")
            plt.title(f"Multi-User Barcode for {var}")
            plt.xlabel("Date")
            plt.ylabel("User ID")
            plt.tight_layout()

            if save:
                dir.mkdir(parents = True, exist_ok= True)
                save_path = dir / f"multibar_{var}.png"
                plt.savefig(save_path)
                plt.close()


            else:
                plt.show()

    def value_distribution_per_id(self):
        """ Visualize the distribution of values per id. """
        grouped = self.data.groupby('id')
        for id_val, group in grouped:
            plt.figure(figsize=(10, 4))
            plt.hist(group['value'], bins=50, color='green', alpha=0.7)
            plt.xlabel('value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of values for ID {id_val}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
    
    def value_distribution_per_variable(self, type = "hist"):
        """ Visualize the distribution of values per variable"""
        grouped = self.data.groupby('variable')

        if type == "box":
            # Boxplots option might be better, especially for outlier visualization
            pass

        if type == "hist":
            for id_var, group in grouped:
                plt.figure(figsize=(10, 4))
                plt.hist(group['value'], bins=50, color='green', alpha=0.7)
                plt.xlabel('value')
                plt.ylabel('Frequency')
                plt.title(f"Distribution of values for variable {id_var}")
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()

    def variable_distribution_per_id(self):
        """ Visualize the distribution of a specified variable per id. """
        grouped = self.data.groupby('id')
        for id_val, group in grouped:
            plt.figure(figsize=(10, 4))
            plt.hist(group['variable'], bins=50, color='orange', alpha=0.7)
            plt.xlabel('variable')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of variable for ID {id_val}')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

    def visualize_value_distribution_per_variable(self):
        """ Visualize the distribution of values per variable, showing all IDs as overlapping lines. """
        # Group the data by variable first
        grouped_by_var = self.data.groupby('variable')
        
        for var_val, var_group in grouped_by_var:
            # Drop NaNs to keep the visualization clean
            clean_data = var_group.dropna(subset=['value'])
            
            # A standard figure size is best for overlapping lines on a single axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use Seaborn's kdeplot to create overlapping density lines
            sns.kdeplot(
                data=clean_data,
                x='value',
                hue='id',           # Creates a separate line for each ID
                ax=ax,
                palette='Set2',     # A visually pleasing, distinct color palette
                linewidth=2,        # Make the lines slightly thicker for visibility
                common_norm=False   # Scales each ID's curve independently so they are comparable
            )
            
            ax.set_title(f'Value Distribution for Variable: {var_val}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            
            # Show standard grid lines for easier reading
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Despine removes the top and right borders for a cleaner, modern look
            sns.despine()
            
            plt.tight_layout()
            plt.show()

    # --- Visualisation of daily data ---
    def show_correlations(self, save: bool = False):

        dir = Path("results/eda/correlation_matrices")
        dir.mkdir(exist_ok=True, parents=True)

        ind_data = self.daily_data.groupby('id')

        for id, ind in ind_data:
            corr_matrix = _wide_format_daily(data = ind).corr()
            plt.figure(figsize=(12, 5))
            sns.heatmap(
                corr_matrix,
                vmin=-1,
                vmax=1,
                cmap='RdBu')
            plt.title(f"Correlation Matrix of all Variables in Daily Data for {id}")
            if save:    
                plt.savefig(dir / f"{id}.png")
                plt.close()
            else: 
                plt.show()


# === Functions ===
def _wide_format_daily(data: pd.DataFrame):
    return data.pivot(index = ['id', 'date'], columns = 'variable', values= 'value')

