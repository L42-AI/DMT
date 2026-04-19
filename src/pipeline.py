import pandas as pd
import numpy as np
import torch

from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from analyser import Analyser
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, List
from features.features_behavioural import add_step_behavioural_features
from features.feature_pipeline import add_static_temporal_features, add_rolling_history
from models import SimpleMLP, SimpleGRU, RandomClassificationBaseline, RandomRegressionBaseline, XGBoostClassifierWrapper

class BasePipeline:
    CLASSIFICATION: bool
    num_classes: int
    num_ids: int
    
    def __init__(self, analyser, batch_size=32):
        self.analyser = analyser
        self.batch_size = batch_size
        self.num_ids = None
        self.input_dim = None
        self.class_mapping = None


    def _trim_individual_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trims leading empty data and trailing data without targets for each user.
        Assumes df is in wide format with 'id' as a column and 'time' as index.
        """
        trimmed_dfs = []
        
        # Group by individual ID
        for user_id, group in df.groupby('id'):
            values = group.values # Converting to numpy for your logic
            
            # --- Leading Trim ---
            # Find first row with sensor activity (cols 0-13) AND a valid mood (col 16)
            start_idx = 0
            for t in range(values.shape[0]):
                # Check if sensors (first 13 cols) are NOT all zero
                if not np.isnan(values[t, 1:14]).all():
                    # Now find the first valid mood from this point forward
                    found_start = False
                    for t2 in range(t, values.shape[0]):
                        if not np.isnan(values[t2, 17]):
                            start_idx = t2
                            found_start = True
                            break
                    if found_start:
                        break
            
            # --- Trailing Trim ---
            # Find the last index with a valid mood
            end_idx = values.shape[0]
            for t in range(values.shape[0] - 1, start_idx, -1):
                if not np.isnan(values[t, 16]):
                    end_idx = t + 1 # +1 for inclusive slicing
                    break
            
            # Apply the slice to the pandas group
            trimmed_dfs.append(group.iloc[start_idx:end_idx])
        
        return pd.concat(trimmed_dfs).sort_index()



    def _prepare_base_data(self) -> pd.DataFrame:
        """ Pivots the dataframe, resets ID to a column, and sets Time as the sole index. """
        wide_format = self.analyser.data.pivot_table(index=['time', 'id'], columns='variable', values='value')
        wide_format = wide_format.reset_index(level='id')

        wide_format['id'] = wide_format['id'].apply(lambda x: int(str(x)[-2:]))
        
        
        wide_format['id'] = wide_format['id'].astype('category').cat.codes

        wide_format = self._trim_individual_series(wide_format)
        
        return wide_format.sort_index()

    def _split_x_y_id(self, df: pd.DataFrame):
        """ Splits the enriched dataframe into Features (X), Targets (y), and IDs. """
        id_series = df.pop('id')
        self.num_ids = id_series.nunique()
        
        y = df.pop('mood')
        X = df.drop(columns=['circumplex.valence', 'circumplex.arousal'], errors='ignore')
        
        return X, y, id_series

    def _clean_data(self, X_df):
        return X_df
    
    def _impute_multivariate(self, X_train, X_val=None, X_test=None, turn_off = False):
        if turn_off:
            return X_train, X_val, X_test
        # 1. Initialize the MissForest proxy (IterativeImputer with RF)
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, n_jobs=-1),
            max_iter=20,
            random_state=42
        )

        # Helper to flatten 3D -> 2D
        def flatten_if_3d(arr):
            if arr is not None and len(arr.shape) == 3:
                # Reshape to (Samples * Seq_Len, Features)
                return arr.reshape(-1, arr.shape[-1]), arr.shape
            return arr, None

        # Helper to restore 2D -> 3D
        def restore_if_3d(arr, original_shape):
            if original_shape is not None:
                return arr.reshape(original_shape)
            return arr

        # 1. Prepare data for Imputer
        X_train_2d, shape_tr = flatten_if_3d(X_train)
        X_val_2d, shape_val = flatten_if_3d(X_val)
        X_test_2d, shape_test = flatten_if_3d(X_test)

        # 2. Fit and Transform
        # Fit ONLY on train to prevent leakage
        X_train_imputed_2d = imputer.fit_transform(X_train_2d)
        
        # Transform Val and Test (if they exist)
        X_val_imputed_2d = imputer.transform(X_val_2d) if X_val_2d is not None else None
        X_test_imputed_2d = imputer.transform(X_test_2d) if X_test_2d is not None else None

        # 3. Reshape back to original dimensions
        X_train_final = restore_if_3d(X_train_imputed_2d, shape_tr)
        X_val_final = restore_if_3d(X_val_imputed_2d, shape_val)
        X_test_final = restore_if_3d(X_test_imputed_2d, shape_test)

        # --- DIAGNOSTIC CHECK ---
        print("\n" + "-"*30)
        print("🔍 IMPUTATION DIAGNOSIS")
        
        # 1. Check for remaining NaNs (Should be 0)
        total_nans = np.isnan(X_train_final).sum()
        print(f"Total NaNs remaining in X_train: {total_nans}")

        # 2. Check Sensor Integrity (Assuming sensors are early columns)
        # Check if sensor zeros were preserved
        sensor_sample = X_train_final.reshape(-1, X_train_final.shape[-1])[:, :5]
        print(f"Sensor sample (first 5 cols) - Mean: {np.mean(sensor_sample):.4f}")
        
        # 3. Check Self-Report Richness (MissForest Activity)
        # Check the variance of the report variables (e.g., mood lags)
        # If std > 0, MissForest is actually predicting values, not just filling zeros.
        report_sample = X_train_final.reshape(-1, X_train_final.shape[-1])[:, 16] # Use your mood index
        print(f"Report Imputation (Col 16) - Std Dev: {np.std(report_sample):.4f}")
        print("-"*30 + "\n")

        return X_train_final, X_val_final, X_test_final

    def _scale_features(self, X_train, X_val, X_test):
        from sklearn.preprocessing import StandardScaler
        import warnings
        import numpy as np

        scaler = StandardScaler()
        
        # Helper to reshape for scaling
        def _reshape_and_scale(X, is_fit=False):
            if X is None:
                return None
            was_3d = False
            if X.ndim == 3:
                was_3d = True
                B, S, F = X.shape
                X = X.reshape(-1, F)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if is_fit:
                    X_scaled = scaler.fit_transform(X)
                else:
                    X_scaled = scaler.transform(X)
                
            if was_3d:
                X_scaled = X_scaled.reshape(B, S, F)
            return X_scaled

        X_train_scaled = _reshape_and_scale(X_train, is_fit=True) if X_train is not None else None
        X_val_scaled = _reshape_and_scale(X_val, is_fit=False) if X_val is not None else None
        X_test_scaled = _reshape_and_scale(X_test, is_fit=False) if X_test is not None else None
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def _engineer_leaky_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().sort_values(['id', 'time'])
        state_features = ['mood', 'circumplex.valence', 'circumplex.arousal']
        
        for col in state_features:
            if col not in df.columns:
                continue
                
            # 2. Prevent Leakage (Shift first!)
            df[f'prev_{col}'] = df.groupby('id')[col].shift(1)
            df[f'prev_{col}'] = df.groupby('id')[f'prev_{col}'].ffill().fillna(1)

            # 3. Calculate Age (Hours since last report)
            # We find where reports exist
            has_report = df[col].notna()
            
            # Create a 'session' ID that increments every time a report is found
            # This groups all hours following a report together
            report_sessions = has_report.groupby(df['id']).cumsum()
            
            # Now we just count the entries within each session
            # This resets to 0 every time a new report starts a new session
            df[f'{col}_age'] = df.groupby(['id', report_sessions]).cumcount()
            
            # 4. Scale Age to 0-1 (Capped at 24 hours)
            df[f'{col}_age'] = df[f'{col}_age'].clip(upper=24) / 24.0

        return df

    # --- ABSTRACT METHODS ---
    def _engineer_features(self, X_df):
        raise NotImplementedError("Subclasses must define how to engineer features.")
    
    def _build_tensors(self, X_df, y_series, id_series):
        raise NotImplementedError()

    def fit_transform_targets(self, y_train: np.ndarray) -> np.ndarray:
        """ Fits quantile bins strictly on training data to prevent leakage. """
        if not self.CLASSIFICATION:
            self.class_mapping = None
            return y_train

        y_flat = y_train.flatten()

        # ==========================================
        # 🐛 FIX: The Jitter Method to break ties
        # ==========================================
        # Add infinitesimal noise to prevent bin boundaries from colliding on exact duplicates.
        # A fixed random seed ensures this step is 100% reproducible across runs.
        np.random.seed(42) 
        jitter = np.random.normal(loc=0.0, scale=1e-6, size=y_flat.shape)
        y_jittered = y_flat + jitter

        # FIT & TRANSFORM (Training Data Only)
        y_binned, self.bin_edges = pd.qcut(
            y_jittered, q=self.num_classes, labels=False, duplicates='drop', retbins=True
        )
        import matplotlib.pyplot as plt
        plt.hist(y_binned)
        plt.show()
        # # FIT & TRANSFORM (Training Data Only)
        # y_binned, self.bin_edges = pd.qcut(
        #     y_train.flatten(), q=self.num_classes, labels=False, duplicates='drop', retbins=True
        # )
        
        # Replace outer edges with infinity so unseen test extremes don't become NaNs
        self.bin_edges[0] = -float('inf')
        self.bin_edges[-1] = float('inf')

        # Save the centroids for evaluation mapping
        self.class_mapping = pd.Series(y_train.flatten()).groupby(y_binned).mean().to_dict()

        return y_binned

    def transform_targets(self, y_eval: np.ndarray) -> np.ndarray:
        """ Transforms validation/test data using the boundaries learned from train. """
        if not self.CLASSIFICATION:
            return y_eval
            
        if not hasattr(self, 'bin_edges'):
            raise ValueError("You must run fit_transform_targets on training data first!")
        
        # TRANSFORM ONLY (Validation/Test Data)
        y_binned = pd.cut(y_eval.flatten(), bins=self.bin_edges, labels=False, include_lowest=True)
        
        return y_binned

    def get_walk_forward_loaders(
            self, 
            n_splits: int = 5, 
            gap: int = 2,
            test_ratio: float = 0.15, 
            tabular: bool = False
            ) -> Tuple[List, DataLoader] | Tuple[List, dict]:
        """ Provides walk-forward folds and a final holdout test set.

        Args:
            n_splits (int, optional): number of splits. Defaults to 5.
            gap (int, optional): gap between training and validation sets. Defaults to 2.
            test_ratio (float, optional): ratio of data to use for testing. Defaults to 0.15.
            tabular (bool, optional): whether to use tabular data format. Defaults to False.

        Returns:
            : _description_
        """
        def package_data(
                X: np.ndarray, 
                y: np.ndarray, 
                id_val: np.ndarray, 
                time_val: np.ndarray, 
                shuffle=False
            ) -> DataLoader | dict:
            """ Depending on the pipeline type, either packages data into PyTorch DataLoaders or returns dicts of NumPy arrays for XGBoost.

            Args:
                X (np.ndarray): Feature array.
                y (np.ndarray): Target array.
                id_val (np.ndarray): ID array.
                time_val (np.ndarray): Time array.
                shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

            Returns:
                DataLoader|dict: Either a PyTorch DataLoader for time-series pipelines or a dict of NumPy arrays for tabular pipelines.
            """
            if tabular:
                return {'X': X, 'y': y}
            
            dataset = TensorDataset(
                torch.tensor(id_val.copy(), dtype=torch.long),
                torch.tensor(X.copy(), dtype=torch.float32),
                torch.tensor(y.copy(), dtype=y_dtype),
                torch.tensor(time_val.copy(), dtype=torch.long)
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
        
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        # Unpack 4 tensors now
        X_arr, y_arr, id_arr, time_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)
        self.input_dim = X_arr.shape[-1]
        
        # Global chronological ordering across all users
        global_order = np.argsort(time_arr, kind='mergesort')
        X_ord = X_arr[global_order]
        y_ord = y_arr[global_order]
        id_ord = id_arr[global_order]
        time_ord = time_arr[global_order]

        # Chronological holdout split
        split_pt = int(len(X_ord) * (1 - test_ratio))

        X_train_full = X_ord[:split_pt]
        y_train_pool = y_ord[:split_pt]
        id_train_pool = id_ord[:split_pt]
        time_train_pool = time_ord[:split_pt]

        X_test_full = X_ord[split_pt:]
        y_test_full = y_ord[split_pt:]
        id_test_full = id_ord[split_pt:]
        time_test_full = time_ord[split_pt:]

        y_train_pool = self.fit_transform_targets(y_train_pool)
        y_test_full = self.transform_targets(y_test_full)

        # Prepare final holdout test set
        X_train_full_imputed, _, X_test_full = self._impute_multivariate(X_train_full, None, X_test_full, turn_off=True)
        _, _, X_test_scaled_full = self._scale_features(X_train_full_imputed, None, X_test_full)
        
        test_data = package_data(
            X_test_scaled_full,
            y_test_full,
            id_test_full,
            time_test_full,
            shuffle=False
        )


        """
        PREVIOUS USER-LEVEL SPLIT (RETAINED FOR REFERENCE)
        """
        # train_idx, test_idx = [], []
        # for uid in np.unique(id_arr):   
        #     idx = np.where(id_arr == uid)[0]
        #     split_pt = int(len(idx) * (1 - test_ratio))
        #     train_idx.extend(idx[:split_pt])
        #     test_idx.extend(idx[split_pt:])
        
        # # Prepare the final holdout test set
        # X_train_full, X_test_full = X_arr[train_idx], X_arr[test_idx]
        # _, _, X_test_scaled_full = self._scale_features(None, None, X_test_full)
        # test_data = package_data(X_test_scaled_full, y_arr[test_idx], id_arr[test_idx], time_arr[test_idx], shuffle=False)
        # Prepare the training pool
        # y_train_pool = y_arr[train_idx]
        # id_train_pool = id_arr[train_idx]
        # time_train_pool = time_arr[train_idx]
        
        """
        END PREVIOUS USER-LEVEL SPLIT (RETAINED FOR REFERENCE)
        """
        
        

        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        folds = []
        for tr_fold_idx, val_fold_idx in tss.split(X_train_full):

            """
                SAFETY CHECK: ENFORCE STRICT TEMPORAL PROTOCOL
                This ensures that the training fold only contains data points that are strictly earlier than any data point in the validation fold, thus preventing any temporal leakage.
            """
            # Safety check: enforce strict forward time order in each fold
            tr_times = time_train_pool[tr_fold_idx]
            val_times = time_train_pool[val_fold_idx]
            if tr_times.max() > val_times.min():
                raise ValueError(
                    "Temporal protocol violation: train fold contains timestamps "
                    "that are not strictly earlier than validation fold."
                )

            """
                END SAFETY CHECK
            """

            # Extract fold-specific training and validation sets
            X_tr_fold, X_val_fold = X_train_full[tr_fold_idx], X_train_full[val_fold_idx]

            # Independently scale them to prevent data leakage
            X_tr_fold, X_val_fold, _ = self._impute_multivariate(X_tr_fold, X_val_fold, None, turn_off=True)
            x_tr_fold_scaled, x_val_fold_scaled, _= self._scale_features(X_tr_fold, X_val_fold, None)

            # Package into DataLoaders, or dicts if tabular
            train_out = package_data(x_tr_fold_scaled, y_train_pool[tr_fold_idx], id_train_pool[tr_fold_idx], time_train_pool[tr_fold_idx], shuffle=not tabular)
            val_out = package_data(x_val_fold_scaled, y_train_pool[val_fold_idx], id_train_pool[val_fold_idx], time_train_pool[val_fold_idx], shuffle=False)

            # Add fold data to list.
            folds.append((train_out, val_out) if not tabular else {'train': train_out, 'val': val_out})
        
        return folds, test_data
    
    def get_dataloaders(self, train_ratio=0.7, val_ratio=0.15, as_numpy=False):
        df = self._prepare_base_data()
        df = self._clean_data(df)
        df = self._engineer_features(df)

        X_df, y_series, id_series = self._split_x_y_id(df)

        # Unpack 4 tensors now
        X_arr, y_arr, id_arr, time_arr, y_dtype = self._build_tensors(X_df, y_series, id_series)

        self.input_dim = X_arr.shape[-1]

        # 6. Global Chronological Split PER USER
        train_idx, val_idx, test_idx = [], [], []

        for uid in np.unique(id_arr):
            idx = np.where(id_arr == uid)[0]
            n_user = len(idx)
            train_end = int(n_user * train_ratio)
            val_end = int(n_user * (train_ratio + val_ratio))
            
            train_idx.extend(idx[:train_end])
            val_idx.extend(idx[train_end:val_end])
            test_idx.extend(idx[val_end:])

        # Apply indices to all arrays including time
        X_train, y_train, id_train, time_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx], time_arr[train_idx]
        X_val, y_val, id_val, time_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx], time_arr[val_idx]
        X_test, y_test, id_test, time_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx], time_arr[test_idx]

        y_train = self.fit_transform_targets(y_train)
        y_val = self.transform_targets(y_val)
        y_test = self.transform_targets(y_test)

        X_train, X_val, X_test = self._scale_features(X_train, X_val, X_test)

        # 8. Package into PyTorch DataLoaders
        def make_loader(X_slice, y_slice, id_slice, time_slice, shuffle):
            dataset = TensorDataset(
                torch.tensor(id_slice.copy(), dtype=torch.long),
                torch.tensor(X_slice.copy(), dtype=torch.float32),
                torch.tensor(y_slice.copy(), dtype=y_dtype),
                torch.tensor(time_slice.copy(), dtype=torch.long) # Added Time Tensor
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)

        train_loader = make_loader(X_train, y_train, id_train, time_train, shuffle=True)
        val_loader = make_loader(X_val, y_val, id_val, time_val, shuffle=False)
        test_loader = make_loader(X_test, y_test, id_test, time_test, shuffle=False)

        print(f"\n--- {self.__class__.__name__} Ready ---")
        print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    

class TabularPipeline(BasePipeline):
    def __init__(self, analyser: Analyser, batch_size: int = 32, lookahead: int = 1, windows: list[int] = [3, 5]):
        """ Pipeline for tabular data with a lookahead mechanism for future target prediction.

        Args:
            analyser (Analyser): Data analyser instance containing the raw data.
            batch_size (int, optional): Batch size for the data loaders. Defaults to 32.
            lookahead (int, optional): Lookahead value for future target prediction. Defaults to 1.
            windows (list[int], optional): List of window sizes for rolling features. Defaults to [3, 5].
        """
        super().__init__(analyser, batch_size)
        self.lookahead = lookahead
        self.windows = windows

    @property
    def max_memory(self):
        """ Calculates the maximum temporal lookback of this pipeline. """
        return max(self.windows)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        
        cols_lag_mean = ['work_leisure_ratio', 'activity']
        cols_sum = ['screen', 'call', 'sms']
        df = add_rolling_history(df, cols_lag_mean, cols_sum, windows=self.windows)
        
        return df
    
    def _build_tensors(self, X_df, y_series, id_series):
        # 1. Shift target backwards by 'lookahead'
        y_future = y_series.groupby(id_series).shift(-self.lookahead)
        
        # Also shift the index (time) backwards so we map the exact future timestamp
        current_times = pd.Series(y_series.index, index=y_series.index)
        time_future = current_times.groupby(id_series).shift(-self.lookahead)
        
        # 2. VALID-TARGET MASK
        valid_mask = ~y_future.isna()
        
        # 3. Filter X, y, IDs, and Times simultaneously
        X_valid = X_df[valid_mask]
        y_valid = y_future[valid_mask]
        id_valid = id_series[valid_mask]
        time_valid = time_future[valid_mask]
        
        y_dtype = torch.long if self.CLASSIFICATION else torch.float32

        # 5. Convert to Numpy arrays
        # DEBUG
        # X_arr = np.nan_to_num(X_valid.values, nan=0.0)
        X_arr = X_valid.values
        y_arr = y_valid.values
        id_arr = id_valid.values
        
        # Convert Pandas Timestamps to Unix Seconds
        time_arr = (time_valid.astype('int64') // 10**9).values
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] 
            
        return X_arr, y_arr, id_arr, time_arr, y_dtype


    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleMLP(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_ids=self.num_ids, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )

    def build_xgboost_model(self, **kwargs):
        """Builds an XGBoost classifier with a wrapper."""
        # Default XGBoost parameters for classification
        if self.num_classes == 2:
            defaults = {
                'n_estimators': 100,
                'learning_rate': 0.001,
                'max_depth': 3,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
            }
        else:
            defaults = {
                'n_estimators': 100,
                'learning_rate': 0.001,
                'max_depth': 3,
                'objective': 'multi:softmax',
                'num_class': int(self.num_classes),
                'eval_metric': 'mlogloss',
            }
        # Override defaults with any user-provided kwargs
        defaults.update(kwargs)
        
        return XGBoostClassifierWrapper(**defaults)

class TimeSeriesPipeline(BasePipeline):
    def __init__(self, analyser, seq_len=7, batch_size=32):
        super().__init__(analyser, batch_size)
        self.seq_len = seq_len

    @property
    def max_memory(self):
        return self.seq_len

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_step_behavioural_features(df)
        df = add_static_temporal_features(df)
        df = self._engineer_leaky_covariates(df)
        return df 
    
    def _build_tensors(self, X_df, y_series, id_series):
        y_dtype = torch.long if self.CLASSIFICATION else torch.float32

        seq_X, seq_y, seq_ids, seq_times = [], [], [], []
        time_index = y_series.index
        
        for user_id in id_series.unique():
            mask = (id_series == user_id)
            user_X = X_df[mask].values
            user_y = y_series[mask].values
            user_times = time_index[mask] 
            
            for i in range(len(user_X) - self.seq_len):
                target_mood = user_y[i + self.seq_len]
                target_time = user_times[i + self.seq_len]
                
                if not np.isnan(target_mood):
                    seq_X.append(user_X[i : i + self.seq_len])
                    seq_y.append(target_mood)
                    seq_ids.append(user_id)
                    seq_times.append(int(target_time.timestamp()))
                    
        X_arr = np.array(seq_X)
        y_arr = np.array(seq_y)
        id_arr = np.array(seq_ids)
        time_arr = np.array(seq_times)
        
        if not self.CLASSIFICATION:
            y_arr = y_arr[:, np.newaxis] 
            
        return X_arr, y_arr, id_arr, time_arr, y_dtype

    def build_model(self, hidden_dim: int = 64, embed_dim: int = 5, dropout_rate: float = 0.5):
        return SimpleGRU(
            input_dim=self.input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=self.num_classes, 
            num_ids=self.num_ids, 
            embed_dim=embed_dim,
            dropout_rate=dropout_rate
        )


class ClassificationPipelineMixin:
    num_ids: int
    CLASSIFICATION = True
    def __init__(self, *args, num_classes=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def build_baseline_model(self):
        return RandomClassificationBaseline(
            output_dim=self.num_classes, 
            num_ids=self.num_ids, 
            embed_dim=5,
        )
    
class RegressionPipelineMixin:
    num_ids: int
    CLASSIFICATION = False
    num_classes = 1

    def build_baseline_model(self):
        return RandomRegressionBaseline(
            output_dim=self.num_classes, 
            num_ids=self.num_ids, 
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
