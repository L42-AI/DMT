import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit

from features.features_behavioural import add_step_behavioural_features
from features.feature_pipeline import add_static_temporal_features, add_rolling_history

from models import SimpleMLP, SimpleGRU, RandomClassificationBaseline, RandomRegressionBaseline

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
        self.num_users = None
        self.input_dim = None

    def _prepare_base_data(self) -> pd.DataFrame:
        """ Pivots the dataframe, resets ID to a column, and sets Time as the sole index. """
        # Put 'time' first so it becomes the primary index when 'id' is reset
        wide_format = self.analyser.data.pivot_table(index=['time', 'id'], columns='variable', values='value')
        
        # Move 'id' to a standard column
        wide_format = wide_format.reset_index(level='id')
        
        # Clean the ID strings into integers (e.g. 'AS14.01' -> 1)
        wide_format['id'] = wide_format['id'].apply(lambda x: int(str(x)[-2:]))
        wide_format['id'] = wide_format['id'].astype('category').cat.codes
        
        # Sort globally by time
        return wide_format.sort_index()

    def _split_x_y_id(self, df: pd.DataFrame):
        """ Splits the enriched dataframe into Features (X), Targets (y), and IDs. """
        id_series = df.pop('id')
        y = df.pop('mood')
        
        X = df.drop(columns=['circumplex.valence', 'circumplex.arousal'], errors='ignore') # TODO: EXAMINE HOW WE CAN USE THESE
        
        # Track total unique IDs to size the embedding layer correctly
        self.num_users = id_series.nunique()

        return X, y, id_series

    def _clean_data(self, X_df):
        return X_df

    def _scale_features(self, X_train, X_val, X_test):
        return X_train, X_val, X_test

    # --- ABSTRACT METHODS (To be overridden by children) ---
    def _engineer_features(self, X_df):
        raise NotImplementedError("Subclasses must define how to engineer features.")
    
    def _process_features(self, X_df, id_series):
        raise NotImplementedError("Subclasses must define how to shape features (Tabular vs TimeSeries).")

    def _process_targets(self, y_series):
        raise NotImplementedError("Subclasses must define how to shape targets (Regression vs Classification).")

    # --- OPTIONAL OVERRIDE: Walk-Forward Splits for Time Series ---
    def get_walk_forward_loaders(self, n_splits=5, gap=2, test_ration=0.15):
        """
        Creates walk-forward splits for time series data. Output is a list of (train_loader, val_loader) tuples for each fold, plus a separate test_loader.
        The test set is the last `test_ratio` portion of the data, and is not included in the walk-forward splits. 
        The walk-forward splits are created on the remaining train+val set.

        args:
        -----
            - n_splits: Number of walk-forward splits to create on the train+val set.
            - gap: Number of time steps to exclude between train and val sets in each fold to prevent leakage.
            - test_ratio: Proportion of the data to reserve as a final test set (default 15%).
        """

        def make_loader(X_slice, y_slice, id_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype)
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        
        X_df, y_series, id_series = self._prepare_base_data()
        
        X_arr, id_arr = self._process_features(X_df, id_series)
        y_arr, y_dtype = self._process_targets(y_series, id_series)
   
        # Split into train+val and test based on time (last 15% for testing)
        n = len(X_arr)
        test_start = int(n * (1 - test_ration))
        X_train_val, y_train_val, id_train_val = X_arr[:test_start], y_arr[:test_start], id_arr[:test_start]
        X_test, y_test, id_test = X_arr[test_start:], y_arr[test_start:], id_arr[test_start:]

        # Create walk-forward splits on the train+val set
        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        folds = []
        for train_index, val_index_index in tss.split(X_train_val):
            train_loader = make_loader(X_train_val[train_index], y_train_val[train_index], id_train_val[train_index], shuffle=True)
            val_loader = make_loader(X_train_val[val_index_index], y_train_val[val_index_index], id_train_val[val_index_index], shuffle=False)
            folds.append((train_loader, val_loader))
        test_loader = make_loader(X_test, y_test, id_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")

        return folds, test_loader 
    
    # --- THE TEMPLATE METHOD ---
    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15):
        # 1. Pivot and sort chronologically (keeping 'time' as the index and 'id' as a column)
        df = self._prepare_base_data()

        # 2. Universal Data Cleaning (e.g., handling missing raw sensor data)
        df = self._clean_data(df)

        # 3. Architecture-Specific Feature Engineering (Delegated to Tabular or TimeSeries child class)
        df = self._engineer_features(df)

        # 4. Late Splitting: Separate Features (X), Targets (y), and IDs
        X_df, y_series, id_series = self._split_x_y_id(df)

        # 5. Shape into Arrays/Sequences based on the architecture
        X_arr, id_arr = self._process_features(X_df, id_series)
        y_arr, y_dtype = self._process_targets(y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split (No data leakage!)
        n = len(X_arr)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        X_train, y_train, id_train = X_arr[:train_end], y_arr[:train_end], id_arr[:train_end]
        X_val, y_val, id_val = X_arr[train_end:val_end], y_arr[train_end:val_end], id_arr[train_end:val_end]
        X_test, y_test, id_test = X_arr[val_end:], y_arr[val_end:], id_arr[val_end:]

        # 7. Secure Feature Scaling (Fit on Train, Transform Val and Test)
        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)

        # 8. Package into PyTorch DataLoaders
        def make_loader(X_slice, y_slice, id_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype)
            )
            # Only drop the last batch if we actually have enough data to form full batches
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

        train_loader = make_loader(X_train, y_train, id_train, shuffle=True)
        val_loader = make_loader(X_val, y_val, id_val, shuffle=False)
        test_loader = make_loader(X_test, y_test, id_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")
        print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
class TabularPipeline(BasePipeline):
    """ Intermediate class for 2D Data (Decision Trees / Simple NNs). """
    def _process_features(self, X_df, id_series):
        return X_df.values, id_series.values
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tabular models lack memory. They need BOTH static features and 
        explicit rolling history (moving averages) engineered into the rows.
        """
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        
        cols_lag_mean = ['work_leisure_ratio', 'activity']
        cols_sum = ['screen', 'call', 'sms']
        df = add_rolling_history(df, cols_lag_mean, cols_sum, windows=[3, 7])
        
        return df.dropna()

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
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        return df.dropna()
    

class TabularClassification(TabularPipeline):
    def __init__(self, analyser, num_bins=5, batch_size=32):
        super().__init__(analyser, batch_size)
        self.num_bins = num_bins
        self.num_classes = None

    def _process_targets(self, y_series, id_series):
        binned_y = pd.qcut(y_series, q=self.num_bins, labels=False)
        self.num_classes = len(np.unique(binned_y.dropna()))
        return binned_y.values, torch.long

    def build_baseline_model(self):
        return RandomClassificationBaseline(
            input_dim=self.input_dim, 
            hidden_dim=64, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
            dropout_rate=0
        )

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleMLP(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

class TimeSeriesClassification(TimeSeriesPipeline):
    def __init__(self, analyser, seq_len=7, num_bins=5, batch_size=8):
        super().__init__(analyser, seq_len, batch_size)
        self.num_bins = num_bins
        self.num_classes = None

    def _process_targets(self, y_series, id_series): 
        binned_y = pd.qcut(y_series, q=self.num_bins, labels=False)
        self.num_classes = len(np.unique(binned_y.dropna()))
        
        seq_y = []
        for user_id in id_series.unique():
            user_y = binned_y[id_series == user_id].values
            for i in range(len(user_y) - self.seq_len + 1):
                seq_y.append(user_y[i + self.seq_len - 1])
                
        return np.array(seq_y), torch.long


    def build_baseline_model(self):        
        return RandomClassificationBaseline(
            input_dim=self.input_dim, 
            hidden_dim=64, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
            dropout_rate=0
        )

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleGRU(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

class TabularRegression(TabularPipeline):
    def _process_targets(self, y_series, id_series):
        return y_series.values[:, np.newaxis], torch.float32

    def build_baseline_model(self):
        return RandomRegressionBaseline(
            input_dim=self.input_dim, 
            hidden_dim=64, 
            output_dim=1, 
            num_users=self.num_users, 
            embed_dim=5,
            dropout_rate=0
        )

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleMLP(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=1, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

class TimeSeriesRegression(TimeSeriesPipeline):
    def _process_targets(self, y_series, id_series):
        seq_y = []
        for user_id in id_series.unique():
            user_y = y_series[id_series == user_id].values
            for i in range(len(user_y) - self.seq_len + 1):
                seq_y.append(user_y[i + self.seq_len - 1])
                
        return np.array(seq_y)[:, np.newaxis], torch.float32

    def build_baseline_model(self):
        return RandomRegressionBaseline(
            input_dim=self.input_dim, 
            hidden_dim=64, 
            output_dim=1, 
            num_users=self.num_users, 
            embed_dim=5,
            dropout_rate=0
        )

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleGRU(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=1, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )