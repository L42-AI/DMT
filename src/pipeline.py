import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

__all__ = [
    'TabularClassification',
    'TimeSeriesClassification',
    'TabularRegression',
    'TimeSeriesRegression'
]

class BasePipeline:
    def __init__(self, analyser, batch_size=32):
        self.analyser = analyser
        self.batch_size = batch_size

    def _prepare_base_data(self):
        """ Pivots and sorts the dataframe chronologically. """
        wide_format = self.analyser.data.pivot_table(index=['id', 'time'], columns='variable', values='value')
        
        wide_format['id_col'] = wide_format.index.get_level_values(0)
        wide_format['id_col'] = wide_format['id_col'].apply(lambda x: int(x[-2:]))
        wide_format = wide_format.droplevel(0).sort_index()

        id_col = wide_format.pop('id_col')
        X = wide_format.drop(columns=['mood', 'circumplex.valence', 'circumplex.arousal'])
        y = wide_format['mood']
        
        return X, y, id_col

    # --- ABSTRACT METHODS (To be overridden by children) ---
    def _process_features(self, X_df, id_series):
        raise NotImplementedError("Subclasses must define how to shape features (Tabular vs TimeSeries).")

    def _process_targets(self, y_series):
        raise NotImplementedError("Subclasses must define how to shape targets (Regression vs Classification).")

    # --- THE TEMPLATE METHOD ---
    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15):
        X_df, y_series, id_series = self._prepare_base_data()

        X_arr, id_arr = self._process_features(X_df, id_series)
        # UPDATE: Pass id_series here
        y_arr, y_dtype = self._process_targets(y_series, id_series)

        n = len(X_arr)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        def make_loader(X_slice, y_slice, id_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype)
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

        train_loader = make_loader(X_arr[:train_end], y_arr[:train_end], id_arr[:train_end], shuffle=True)
        val_loader = make_loader(X_arr[train_end:val_end], y_arr[train_end:val_end], id_arr[train_end:val_end], shuffle=False)
        test_loader = make_loader(X_arr[val_end:], y_arr[val_end:], id_arr[val_end:], shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")
        print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
        return train_loader, val_loader, test_loader
    
class TabularPipeline(BasePipeline):
    """ Intermediate class for 2D Data (Decision Trees / Simple NNs). """
    def _process_features(self, X_df, id_series):
        return X_df.values, id_series.values

class TimeSeriesPipeline(BasePipeline):
    """ Intermediate class for 3D Sequence Data (GRU / Transformers). """
    def __init__(self, analyser, seq_len=7, batch_size=32):
        super().__init__(analyser, batch_size)
        self.seq_len = seq_len

    def _process_features(self, X_df, id_series):
        seq_X, seq_ids = [], []
        
        for user_id in id_series.unique():
            mask = (id_series == user_id)
            user_X = X_df[mask].values
            
            for i in range(len(user_X) - self.seq_len + 1):
                seq_X.append(user_X[i : i + self.seq_len])
                seq_ids.append(user_id)
                
        return np.array(seq_X), np.array(seq_ids)
    

class TabularClassification(TabularPipeline):
    def __init__(self, analyser, num_bins=5, batch_size=32):
        super().__init__(analyser, batch_size)
        self.num_bins = num_bins
        self.num_classes = None

    # UPDATE: Add id_series to signature (even though it isn't used for windowing here)
    def _process_targets(self, y_series, id_series):
        binned_y = pd.qcut(y_series, q=self.num_bins, labels=False)
        self.num_classes = len(np.unique(binned_y.dropna()))
        return binned_y.values, torch.long

class TimeSeriesClassification(TimeSeriesPipeline):
    def __init__(self, analyser, seq_len=7, num_bins=5, batch_size=32):
        super().__init__(analyser, seq_len, batch_size)
        self.num_bins = num_bins
        self.num_classes = None

    # UPDATE: Add id_series to signature
    def _process_targets(self, y_series, id_series): 
        binned_y = pd.qcut(y_series, q=self.num_bins, labels=False)
        self.num_classes = len(np.unique(binned_y.dropna()))
        
        seq_y = []
        # UPDATE: Use the perfectly aligned id_series passed from the base pipeline
        for user_id in id_series.unique():
            user_y = binned_y[id_series == user_id].values
            for i in range(len(user_y) - self.seq_len + 1):
                seq_y.append(user_y[i + self.seq_len - 1])
                
        return np.array(seq_y), torch.long

class TabularRegression(TabularPipeline):
    # UPDATE: Add id_series to signature
    def _process_targets(self, y_series, id_series):
        return y_series.values[:, np.newaxis], torch.float32

class TimeSeriesRegression(TimeSeriesPipeline):
    # UPDATE: Add id_series to signature
    def _process_targets(self, y_series, id_series):
        seq_y = []
        # UPDATE: Use the passed id_series
        for user_id in id_series.unique():
            user_y = y_series[id_series == user_id].values
            for i in range(len(user_y) - self.seq_len + 1):
                seq_y.append(user_y[i + self.seq_len - 1])
                
        return np.array(seq_y)[:, np.newaxis], torch.float32