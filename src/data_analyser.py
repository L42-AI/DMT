from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def load_data(file_path: str | None = None) -> pd.DataFrame:
    """ Load the dataset from a CSV file. """
    if file_path is None:
        file_path = Path("data/dataset_mood_smartphone.csv")
    df = pd.read_csv(file_path, index_col=0)  # Assuming the first column is an index
    df['time'] = pd.to_datetime(df['time'])  # Ensure 'time' column is in datetime format
    return df

class Visualiser:
    def __init__(self, data: pd.DataFrame):
        self.data = data

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

class Analyser:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def design_mat(self):
        self.data['time'] = pd.to_datetime(self.data['time'])
        design_mat = self.data.pivot_table(index = ['id', 'time'],
                                           columns = 'variable',
                                           values = 'value',
                                           aggfunc = 'mean'
                                           )
        print(design_mat.head())

    def descriptives(self):
        ''' Descriptive statistics for all variables of the dataset '''
        print(" === Descriptives ===")
        data_audit = self.data.groupby('variable')['value'].describe()
        print(data_audit, '\n')

    def daily_format(self):
        # In progress
        '''Create an daily format database with appropriate aggregates for each variable. '''

        # create date column for each entry
        self.data['date'] = self.data['time'].dt.date

        # lists of variable names; aggregation for durations should be sum, and for scores should be mean
        sum_vars = [v for v in self.data['variable'].unique() if 'appCat' in v or v in ['screen', 'call', 'sms', 'activity']]
        mean_vars = ['mood', 'circumplex.arousal', 'circumflex.valence']

        # want a table of aggregate values for every combination of id, date, and variables
        sum_mask = self.data['variable'].isin(sum_vars)
        mean_mask = self.data['variable'].isin(mean_vars)
        data_daily_sum = self.data[sum_mask].groupby(['id', 'date', 'variable'])['value'].sum().unstack()
        data_daily_mean = self.data[mean_mask].groupby(['id', 'date', 'variable'])['value'].mean().unstack()
        data_daily = pd.concat([data_daily_mean, data_daily_sum], axis = 1)
        

    
        # input 0s as values for duration variables on inactive days 
        duration_vars = [var for var in data_daily.columns if var in sum_vars]
        data_daily[duration_vars] = data_daily[duration_vars].fillna(0)

        # some sort of imputation technique needed for replacing NAs in score aggregates with informative values.  

        # Current: aggregated data with duration aggregate NAs replaced with 0. 
        print(data_daily.head())


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
