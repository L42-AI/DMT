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
    CLASSIFICATION: bool
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
        self.num_users = id_series.nunique()
        
        y = df.pop('mood')
        X = df.drop(columns=['circumplex.valence', 'circumplex.arousal'], errors='ignore')
        
        return X, y, id_series

    def _clean_data(self, X_df):
        return X_df

    def _scale_features(self, X_train, X_val, X_test):
        return X_train, X_val, X_test

    def _engineer_leaky_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        Converts co-targets (valence/arousal) into safe, lagged features (t-1).
        This must be called during feature engineering to prevent data leakage.
        """
        leaky_columns = ['circumplex.valence', 'circumplex.arousal']
        
        for col in leaky_columns:
            if col in df.columns:
                df[f'{col}_lag1'] = df.groupby('id')[col].shift(1)
                
        # Fill the NaNs created on the first day with a neutral baseline (0.0)
        for col in leaky_columns:
            if f'{col}_lag1' in df.columns:
                df[f'{col}_lag1'] = df[f'{col}_lag1'].fillna(0.0)
                
        return df

    # --- ABSTRACT METHODS (To be overridden by children) ---
    def _engineer_features(self, X_df):
        raise NotImplementedError("Subclasses must define how to engineer features.")
    
    def _build_tensors(self, X_df, y_series, id_series):
        """ Subclasses must define how to shape features and targets simultaneously. """
        raise NotImplementedError()

    # --- OPTIONAL OVERRIDE: Walk-Forward Splits for Time Series ---
    def get_walk_forward_loaders(self, n_splits=5, gap=2, test_ratio=0.15):
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
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        X_arr, y_arr, id_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split (No data leakage!)
        train_idx, val_idx, test_idx = [], [], []

        train_ratio = 1.0 - test_ratio

        for uid in np.unique(id_arr):
            # Find all row indices belonging to this specific user
            idx = np.where(id_arr == uid)[0]
            
            n_user = len(idx)
            train_end = int(n_user * train_ratio)
            val_end = int(n_user * (train_ratio + test_ratio))
            
            # Split chronologically within this user's timeline
            train_idx.extend(idx[:train_end])
            val_idx.extend(idx[train_end:val_end])
            test_idx.extend(idx[val_end:])

        # Apply the indices to the arrays
        X_train, y_train, id_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx]
        X_val, y_val, id_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx]
        X_test, y_test, id_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx]

        # 7. Secure Feature Scaling (Fit on Train, Transform Val and Test)
        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)

        # Create walk-forward splits on the train+val set
        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        folds = []
        for train_index, val_index_index in tss.split(X_val):
            train_loader = make_loader(X_val[train_index], y_val[train_index], id_val[train_index], shuffle=True)
            val_loader = make_loader(X_val[val_index_index], y_val[val_index_index], id_val[val_index_index], shuffle=False)
            folds.append((train_loader, val_loader))
        test_loader = make_loader(X_test, y_test, id_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")

        return folds, test_loader 
    
    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15):
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        X_arr, y_arr, id_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split (No data leakage!)
        train_idx, val_idx, test_idx = [], [], []

        for uid in np.unique(id_arr):
            # Find all row indices belonging to this specific user
            idx = np.where(id_arr == uid)[0]
            
            n_user = len(idx)
            train_end = int(n_user * train_ratio)
            val_end = int(n_user * (train_ratio + val_ratio))
            
            # Split chronologically within this user's timeline
            train_idx.extend(idx[:train_end])
            val_idx.extend(idx[train_end:val_end])
            test_idx.extend(idx[val_end:])

        # Apply the indices to the arrays
        X_train, y_train, id_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx]
        X_val, y_val, id_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx]
        X_test, y_test, id_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx]

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
    def __init__(self, analyser, batch_size=32, lookahead=1):
        super().__init__(analyser, batch_size)
        # 'lookahead' dictates how many hours ahead we are predicting.
        # e.g., 1 = next hour, 24 = next day (if data is hourly)
        self.lookahead = lookahead
        
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tabular models lack memory. They need BOTH static features and 
        explicit rolling history (moving averages) engineered into the rows.
        """
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        
        cols_lag_mean = ['work_leisure_ratio', 'activity']
        cols_sum = ['screen', 'call', 'sms']
        df = add_rolling_history(df, cols_lag_mean, cols_sum, windows=[3, 12, 24])
        
        return df
    
    def _build_tensors(self, X_df, y_series, id_series):
        """ Unified builder using pure Vectorization and Valid-Target Masking """
        
        # 1. Shift the target backwards by 'lookahead' steps per user.
        # This aligns the FUTURE mood with the CURRENT row's features.
        y_future = y_series.groupby(id_series).shift(-self.lookahead)
        
        # 2. VALID-TARGET MASK: Find rows where the future mood actually exists
        valid_mask = ~y_future.isna()
        
        # 3. Filter X, y, and IDs simultaneously using the mask
        X_valid = X_df[valid_mask]
        y_valid = y_future[valid_mask]
        id_valid = id_series[valid_mask]
        
        # 4. Handle Classification Binning (applied ONLY to valid targets)
        if self.CLASSIFICATION:
            y_valid = pd.qcut(y_valid, q=self.num_bins, labels=False)
            self.num_classes = len(np.unique(y_valid.dropna()))
            y_dtype = torch.long
        else:
            y_dtype = torch.float32

        # 5. Convert to Numpy arrays
        # np.nan_to_num safely fills any remaining feature NaNs (e.g., from rolling windows) with 0.0
        X_arr = np.nan_to_num(X_valid.values, nan=0.0)
        y_arr = y_valid.values
        id_arr = id_valid.values
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] # Reshape for regression
            
        return X_arr, y_arr, id_arr, y_dtype


    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleMLP(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

class TimeSeriesPipeline(BasePipeline):
    
    def __init__(self, analyser, seq_len=7, batch_size=32):
        super().__init__(analyser, batch_size)
        self.seq_len = seq_len

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        return df 
    
    def _build_tensors(self, X_df, y_series, id_series):
        """ Unified builder with Valid-Target Masking """
        
        # If classification, bin the targets first. pd.qcut safely ignores NaNs!
        if self.CLASSIFICATION:
            y_series = pd.qcut(y_series, q=self.num_bins, labels=False, duplicates='drop')
            self.num_classes = len(np.unique(y_series.dropna()))
            y_dtype = torch.long
        else:
            y_dtype = torch.float32

        seq_X, seq_y, seq_ids = [], [], []
        
        for user_id in id_series.unique():
            mask = (id_series == user_id)
            user_X = X_df[mask].values
            user_y = y_series[mask].values # This array contains many NaNs
            
            # Slide the window hour by hour
            for i in range(len(user_X) - self.seq_len):
                
                # The target is the hour immediately following the sequence window
                target_mood = user_y[i + self.seq_len]
                
                # VALID-TARGET MASK: Only save if the user actually logged their mood!
                if not np.isnan(target_mood):
                    
                    # Optional: You can also check if the input sequence has too many NaNs
                    # if np.isnan(user_X[i : i + self.seq_len]).sum() < some_threshold:
                    
                    seq_X.append(user_X[i : i + self.seq_len])
                    seq_y.append(target_mood)
                    seq_ids.append(user_id)
                    
        # Replace remaining NaNs in the FEATURE sequences with 0s (or your preferred imputation)
        X_arr = np.nan_to_num(np.array(seq_X), nan=0.0)
        y_arr = np.array(seq_y)
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] # Reshape for regression
            
        return X_arr, y_arr, np.array(seq_ids), y_dtype

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleGRU(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

class ClassificationPipelineMixin:
    CLASSIFICATION = True
    num_users: int
    def __init__(self, *args, num_bins=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_bins = num_bins
        self.num_classes = None

    def build_baseline_model(self):
        return RandomClassificationBaseline(
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
        )
    
class RegressionPipelineMixin:
    CLASSIFICATION = False
    num_users: int
    num_classes = 1

    def build_baseline_model(self):
        return RandomRegressionBaseline(
            output_dim=self.num_classes, 
            num_users=self.num_users, 
            embed_dim=5,
        )


class TabularClassification(ClassificationPipelineMixin, TabularPipeline):
    pass

class TimeSeriesClassification(ClassificationPipelineMixin, TimeSeriesPipeline):
    pass

class TabularRegression(RegressionPipelineMixin, TabularPipeline):
    pass

class TimeSeriesRegression(RegressionPipelineMixin, TimeSeriesPipeline):
    pass