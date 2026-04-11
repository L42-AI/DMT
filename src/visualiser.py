from pathlib import Path
from typing import TYPE_CHECKING, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from consts import SRC_DIR

import helpers as _helpers

if TYPE_CHECKING:
    from analyser import Analyser

class Visualiser:
    def __init__(self, data: pd.DataFrame):
        self.load(data)

    def load(self, data: pd.DataFrame, daily_data: pd.DataFrame = None):
        self.data = data
        self.daily_data = daily_data
        if 'date' not in self.data.columns:
            self.data['date'] = self.data['time'].dt.date

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

    def heatmap_missing_values_per_id(self, date: bool = False, save: bool = False):
        """Visualize the presence of NAs amongs individual/variable or individual/date combinations

        Args:
            date (bool, optional): Whether you want to visualize NAs in dates instead of variables. Defaults to False.
            save (bool, optional): Whether you want to save instead of plot. Defaults to False.
        """
        dir = SRC_DIR / "results" / "eda"

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

    def show_na_distribution(self, save: bool = False):
        """ Show the distribution of NA durations for the original data. In hours """
        duration_in_hours = self.data.loc[self.data['gap_duration'] > pd.Timedelta(0),'gap_duration'].dt.total_seconds() / 3600
        plt.hist(duration_in_hours)
        if save:
            plt.savefig("results/eda/na_distribution.png")
            plt.close()
        else:
            plt.show()

    def datapoint_counts_per_id(self, save: bool = False):
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
        if save:
            plt.savefig("results/eda/datapoint_counts.png")
            plt.close()
        else:
            plt.show()

    def ts_dist_per_id(self, save: bool = False):
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
            if save:
                plt.savefig(f"results/eda/timestamp_distribution_{id_val}.png")
                plt.close()
            else:
                plt.show()

    def ts_dist_per_var(self, save: bool = False):
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
            if save:
                plt.savefig(f"results/eda/timestamp_distribution_{var_val}.png")
                plt.close()
            else:
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

    def var_dist_per_id(self, save: bool = False):
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
            if save:
                plt.savefig(f"results/eda/variable_distribution_{id_val}.png")
                plt.close()
            else:
                plt.show()

    def val_dist_per_var(self, save: bool = False):
        """ Visualize the distribution of values per variable, showing all IDs as overlapping lines. """
        # Group the data by variable first
        grouped_by_var = self.data.groupby('variable')
        
        for var_val, var_group in grouped_by_var:
            # Drop NaNs to keep the visualization clean
            clean_data = var_group.dropna(subset=['value'])
            
            # A standard figure size is best for overlapping lines on a single axis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Use Seaborn's kdeplot to create overlapping density lines
            sns.histplot(
                data=clean_data,
                x='value',
                hue='id',
                element='step',
                stat='density',
                common_norm=False,
                palette='tab10',
                alpha=0.5,
                ax=ax,
            )
            
            ax.set_title(f'Value Distribution for Variable: {var_val}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Value', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            
            # Show standard grid lines for easier reading
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Despine removes the top and right borders for a cleaner, modern look
            sns.despine()
            
            plt.tight_layout()
            if save:
                plt.savefig(f'results/eda/value_distribution_{var_val}.png')
                plt.close()
            else:
                plt.show()
    
    # --- Visualisation of daily data ---
    def var_correlations_per_id(self, save: bool = False):

        dir = Path("results/eda/correlation_matrices")
        dir.mkdir(exist_ok=True, parents=True)

        ind_data = self.daily_data.groupby('id')

        for id, ind in ind_data:
            corr_matrix = _helpers.wide_format_daily(data = ind).corr()
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
