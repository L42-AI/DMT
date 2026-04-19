import torch
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from pipeline import TimeSeriesClassification, TimeSeriesRegression, TabularClassification, TabularRegression
from sklearn.ensemble import RandomForestRegressor

from trainer import Trainer
from visualiser import Visualiser
from analyser import Analyser

from final_evaluation import evaluate_predictions, plot_prediction_distributions, evaluate_sklearn_predictions

import data as _data

# Data hyperparameters
INTERVAL = 1
UNIT = 'H'
NUM_CLASSES = 3

# Model hyperparameters
SEQ_LEN = int(7 * 1 / INTERVAL) if UNIT == 'D' else int(7 * 24 / INTERVAL)
EMBEDDING_DIM = 8
HIDDEN_DIM = 32
DROP_RATE = 0.7

# Training hyperparameters
LR = 0.001
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-3
EPOCHS = 50

# XGBoost hyperparameters
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.01,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

# Walk-forward validation hyperparameters
N_SPLITS = 5
GAP = 0  # No gap between train and val for now, but can be adjusted to simulate delayed feedback scenarios
TEST_RATIO = 0.2  # Use the last 20% of the data as a final held-out test set after walk-forward splits     

def _extract_numpy_from_loader(loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to convert a DataLoader back to NumPy arrays."""
    all_X, all_y = [], []
    for _, X_tensor, y_tensor, *_ in loader:
        all_X.append(X_tensor.cpu().numpy())
        all_y.append(y_tensor.cpu().numpy())
    
    X_arr = np.concatenate(all_X)
    # Reshape if data is sequential, as XGBoost expects 2D input
    if X_arr.ndim > 2:
        X_arr = X_arr.reshape(X_arr.shape[0], -1)
        
    return X_arr, np.concatenate(all_y)

def prepare_data() -> Analyser:
    print("--- Loading and Processing Raw Data ---")
    
    # Load the data
    data = _data.load()
    visualiser = Visualiser(data)
    analyser = Analyser(data)

    # Clean and aggregate
    analyser.process_outliers()
    analyser.aggregate.time_data(interval=INTERVAL, unit=UNIT, inplace=True)
    analyser.aggregate.activity(interval=INTERVAL, unit=UNIT, inplace=True)
    analyser.aggregate.communication_events(interval=INTERVAL, unit=UNIT, inplace=True)
    analyser.aggregate.reported_data(interval=INTERVAL, unit=UNIT, inplace=True)

    analyser.apply_scaling(inplace=True) # Scale before imputation to prevent data leakage from imputed values

    # Save intermediate state
    analyser.data.to_csv('data/aggregated_data.csv', index=False)
    
    # Impute missing values
    # print("Running imputation...")
    # analyser.impute(delete=False, catsi=True, epochs=1)
    # analyser.data.to_csv('data/aggregated_data_after_impute.csv', index=False)

    # Quick diagnostic check
    missing_data = analyser.data.isna().any().sum()
    visualiser.heatmap_missing_values_per_id(title = "Percentage of Missing Values in Individuals, after aggregation", save=True)
    print(f"Columns with missing data after imputation: {missing_data}")
    return analyser

def plot_fold_class_distribution(folds, tabular=True):
    rows = []

    for fold_idx, fold in enumerate(folds, start=1):
        if tabular:
            y_train = np.asarray(fold['train']['y']).reshape(-1).astype(int)
            y_val = np.asarray(fold['val']['y']).reshape(-1).astype(int)
        else:
            train_loader, val_loader = fold
            _, y_train = _extract_numpy_from_loader(train_loader)
            _, y_val = _extract_numpy_from_loader(val_loader)
            y_train = np.asarray(y_train).reshape(-1).astype(int)
            y_val = np.asarray(y_val).reshape(-1).astype(int)

        for split_name, y in [('train', y_train), ('val', y_val)]:
            classes, counts = np.unique(y, return_counts=True)
            for c, n in zip(classes, counts):
                rows.append({
                    'fold': fold_idx,
                    'split': split_name,
                    'class': int(c),
                    'count': int(n),
                })

    dist_df = pd.DataFrame(rows)

    # Print missing classes per fold
    all_classes = sorted(dist_df['class'].unique().tolist())
    print("All classes observed across all folds:", all_classes)

    for fold_idx in sorted(dist_df['fold'].unique()):
        for split_name in ['train', 'val']:
            present = sorted(
                dist_df[(dist_df['fold'] == fold_idx) & (dist_df['split'] == split_name)]['class']
                .unique().tolist()
            )
            missing = sorted(list(set(all_classes) - set(present)))
            print(f"Fold {fold_idx} | {split_name} present={present} missing={missing}")

    # Plot
    n_folds = dist_df['fold'].nunique()
    fig, axes = plt.subplots(n_folds, 1, figsize=(10, 3 * n_folds), sharex=True)
    if n_folds == 1:
        axes = [axes]

    for i, fold_idx in enumerate(sorted(dist_df['fold'].unique())):
        ax = axes[i]
        sub = dist_df[dist_df['fold'] == fold_idx]

        pivot_train = sub[sub['split'] == 'train'].set_index('class')['count']
        pivot_val = sub[sub['split'] == 'val'].set_index('class')['count']

        classes = sorted(sub['class'].unique().tolist())
        x = np.arange(len(classes))
        w = 0.36

        train_counts = [pivot_train.get(c, 0) for c in classes]
        val_counts = [pivot_val.get(c, 0) for c in classes]

        ax.bar(x - w/2, train_counts, width=w, label='train')
        ax.bar(x + w/2, val_counts, width=w, label='val')

        ax.set_title(f'Fold {fold_idx} class distribution')
        ax.set_ylabel('count')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()

    axes[-1].set_xlabel('class')
    plt.tight_layout()
    plt.show()

def get_tabular_numpy_splits(pipeline: TabularRegression, train_ratio=0.7, val_ratio=0.15):
    """
    Uses the existing TabularRegression pipeline to prepare engineered tabular features,
    then returns numpy arrays instead of PyTorch dataloaders.
    """
    # Reuse your existing pipeline logic
    df = pipeline._prepare_base_data()
    df = pipeline._clean_data(df)
    df = pipeline._engineer_features(df)

    X_df, y_series, id_series = pipeline._split_x_y_id(df)
    X_arr, y_arr, id_arr, time_arr, _ = pipeline._build_tensors(X_df, y_series, id_series)

    # y comes out as shape [n, 1] for regression; sklearn wants [n]
    y_arr = y_arr.ravel()

    train_idx, val_idx, test_idx = [], [], []

    for uid in np.unique(id_arr):
        idx = np.where(id_arr == uid)[0]
        n_user = len(idx)
        train_end = int(n_user * train_ratio)
        val_end = int(n_user * (train_ratio + val_ratio))

        train_idx.extend(idx[:train_end])
        val_idx.extend(idx[train_end:val_end])
        test_idx.extend(idx[val_end:])

    X_train, y_train, id_train, time_train = X_arr[train_idx], y_arr[train_idx], id_arr[train_idx], time_arr[train_idx]
    X_val, y_val, id_val, time_val = X_arr[val_idx], y_arr[val_idx], id_arr[val_idx], time_arr[val_idx]
    X_test, y_test, id_test, time_test = X_arr[test_idx], y_arr[test_idx], id_arr[test_idx], time_arr[test_idx]

    return {
        "pipeline": pipeline,
        "X_train": X_train, "y_train": y_train, "id_train": id_train, "time_train": time_train,
        "X_val": X_val, "y_val": y_val, "id_val": id_val, "time_val": time_val,
        "X_test": X_test, "y_test": y_test, "id_test": id_test, "time_test": time_test,
    }

def train_classification_model(analyser, save_plotting: bool = False):
    print("\n--- Initializing Machine Learning Pipeline ---")
    
    pipeline = TimeSeriesClassification(analyser, seq_len=SEQ_LEN, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    sample_id, sample_X, sample_y, sample_time = next(iter(train_loader))
    batch, seq_len, num_features = sample_X.shape
    
    print(f"Detected {num_features} input features.")

    # 2. Instantiate the Model
    model = pipeline.build_model(hidden_dim=HIDDEN_DIM, embed_dim=EMBEDDING_DIM, dropout_rate=DROP_RATE)

    # 3. Define Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Train the Model
    trainer = Trainer(model, optimizer, criterion, task_type='classification')
    trainer.fit(train_loader, val_loader=val_loader, num_epochs=EPOCHS, save_history=save_plotting)
    
    results_df, final_mse, final_mae = evaluate_predictions(
        analyser=analyser,
        model=trainer.model, 
        dataloader=test_loader, 
        device=trainer.device,
        class_mapping=pipeline.class_mapping
    )

    plot_prediction_distributions(results_df, resolution=UNIT, save=save_plotting)

def train_regression_model(analyser, save_plotting: bool = False):
    print("\n--- Initializing Machine Learning Pipeline ---")
    
    pipeline = TimeSeriesRegression(analyser, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    sample_id, sample_X, sample_y, sample_time = next(iter(train_loader))
    batch, seq_len, num_features = sample_X.shape
    
    print(f"Detected {num_features} input features.")

    # 2. Instantiate the Model
    model = pipeline.build_model(hidden_dim=HIDDEN_DIM, embed_dim=EMBEDDING_DIM, dropout_rate=DROP_RATE)

    # 3. Define Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = torch.nn.MSELoss()

    # 4. Train the Model on Hourly Data
    trainer = Trainer(model, optimizer, criterion, task_type='regression')
    trainer.fit(train_loader, val_loader=val_loader, num_epochs=EPOCHS, save_history=save_plotting)

    results_df, final_mse, final_mae = evaluate_predictions(
        analyser=analyser,
        model=trainer.model, 
        dataloader=test_loader, 
        device=trainer.device,
        class_mapping=pipeline.class_mapping
    )

    plot_prediction_distributions(results_df, resolution=UNIT, save=save_plotting)

def train_random_forest_regression(analyser, save_plotting: bool = False):
    print("\n--- Initializing Random Forest Regression Pipeline ---")

    pipeline = TabularRegression(analyser, batch_size=BATCH_SIZE, lookahead=24)

    data = get_tabular_numpy_splits(pipeline)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    id_test = data["id_test"]
    time_test = data["time_test"]

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    # Optional validation check
    val_preds = rf.predict(X_val)
    val_mse = mean_squared_error(y_val, val_preds)
    val_mae = mean_absolute_error(y_val, val_preds)

    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")

    # Final test predictions
    test_preds = rf.predict(X_test)

    results_df, final_mse, final_mae = evaluate_sklearn_predictions(
        analyser=analyser,
        preds=test_preds,
        # y_true=y_test,
        ids=id_test,
        times=time_test,
        model_name="Random Forest",
        class_mapping=pipeline.class_mapping
    )

    plot_prediction_distributions(results_df, resolution=UNIT, save=save_plotting)

    return rf, final_mse, final_mae

def walk_forward_train(analyser, tabular=False):
    """ 
    Train a model using walk-forward validation. Exclusively uses walk-forward loader functions.
    Simulates real-world scenario of training on past data and validating on future data.
    
    Args:
        analyser: The Analyser object containing the processed data.
        tabular: If True, uses the Tabular pipeline and XGBoost. If False, uses the TimeSeries pipeline and PyTorch models.
    """
    # 1. Setup Pipeline
    # Using a shorter seq_len as discussed to preserve data points
    if tabular:
        pipeline = TabularClassification(analyser, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, windows=[3, 5])
    else:
        pipeline = TimeSeriesClassification(analyser, seq_len=SEQ_LEN, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    
    # get_walk_forward_loaders returns (folds, test_loader)
    # if tabular, folds is a list of dicts with 'train' and 'val' keys containing NumPy arrays
    # if time-series, folds is a list of tuples (train_loader, val_loader)
    folds, test_data = pipeline.get_walk_forward_loaders(n_splits=N_SPLITS, gap=GAP, test_ratio=TEST_RATIO, tabular=tabular)

    # DEBUG: Print class distribution for each fold to check for missing classes
    # plot_fold_class_distribution(folds, tabular=tabular)

    # 2. Walk-Forward Loop
    fold_results = []
    trained_model = None  # To keep track of the most recently trained model for final evaluation

    for fold_idx, fold_content in enumerate(folds):
        print(f"\n" + "="*30)
        print(f"🚀 STARTING FOLD {fold_idx + 1}/{len(folds)}")
        print(f"="*30)
        
        if tabular:
            # Extract the NumPy arrays from the dictionary for XGBoost
            train_set = fold_content['train']
            val_set = fold_content['val']

            # Build and train the XGBoost model on the current fold
            model = pipeline.build_xgboost_model(**XGBOOST_PARAMS)
            print(f"Fitting XGBoost on {len(train_set['X'])} training samples...")
            model.fit(
                train_set['X'], 
                train_set['y'], 
                eval_set=[(val_set['X'], val_set['y'])], 
                verbose=False
            )

            # print accuracy 
            val_preds = model.predict(val_set['X'])
            val_acc = (val_preds == val_set['y']).mean()
            fold_results.append({'acc': val_acc})
            print(f"Fold {fold_idx + 1} Validation Accuracy: {val_acc:.2%}")

            # Keep the trained model for final evaluation on the held-out test set
            trained_model = model
            
        else:
            # UPDATE: fold_content is a tuple: (train_loader, val_loader)
            train_loader, val_loader = fold_content
            
            # Build and train the PyTorch model on the current fold
            model = pipeline.build_model(hidden_dim=HIDDEN_DIM, embed_dim=EMBEDDING_DIM, dropout_rate=DROP_RATE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            criterion = torch.nn.CrossEntropyLoss()
            trainer = Trainer(model, optimizer, criterion, task_type='classification')
            trainer.fit(train_loader=train_loader, val_loader=val_loader, num_epochs=EPOCHS)
            
            # Evaluate loss and accuracy on the validation set for this fold
            val_metrics = trainer._run_epoch(val_loader, is_train=False)
            fold_results.append(val_metrics)

            # Keep the trained model for final evaluation on the held-out test set
            trained_model = trainer 

    # 3. Final Evaluation
    print("\n" + "X"*40)
    print("      FINAL WALK-FORWARD TEST")
    print("X"*40)

    if tabular:
        # UPDATE: test_data is a dict {'X': ..., 'y': ...}
        test_preds = trained_model.predict(test_data['X'])
        test_acc = (test_preds == test_data['y']).mean()
        test_metrics = {'acc': test_acc}
    else:
        # test_data is a DataLoader
        test_metrics = trained_model._run_epoch(test_data, is_train=False)
    
    avg_fold_acc = sum(f['acc'] for f in fold_results) / len(fold_results)
    
    print(f"Average Fold Validation Accuracy: {avg_fold_acc:.2%}")
    print(f"Final Held-out Test Accuracy:    {test_metrics.get('acc', 0.0):.2%}")

    return fold_results, test_metrics

def train_model(analyser, classification: bool, tabular: bool, seq_len=7, embed_dim=8, windows=[3, 5], save_plotting: bool = False):
    task_str = "Classification" if classification else "Regression"
    mode_str = "Tabular" if tabular else "TimeSeries"
    
    # We suppress prints during tuning so multi-core output doesn't turn into a messy wall of text
    print(f"\n🚀 INITIALIZING PIPELINE: {mode_str} {task_str}")

    # 1. Pipeline Routing (USING THE NEW ARGUMENTS)
    if tabular and classification:
        pipeline = TabularClassification(analyser, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, windows=windows)
    elif tabular and not classification:
        pipeline = TabularRegression(analyser, batch_size=BATCH_SIZE, lookahead=24, windows=windows)
    elif not tabular and classification:
        pipeline = TimeSeriesClassification(analyser, seq_len=seq_len, num_classes=NUM_CLASSES, batch_size=BATCH_SIZE)
    else:
        pipeline = TimeSeriesRegression(analyser, seq_len=seq_len, batch_size=BATCH_SIZE)

    # ==========================================
    # ROUTE A: TABULAR (XGBoost / Random Forest)
    # ==========================================
    if tabular:
        # Extract Numpy arrays for Scikit-Learn API
        data = get_tabular_numpy_splits(pipeline)
        
        X_train, y_train = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        if classification:
            print("Model: XGBoost Classifier")
            model = pipeline.build_xgboost_model()
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            test_preds = model.predict(X_test)
            model_name = "XGBoost"
        else:
            print("Model: Random Forest Regressor")
            model = RandomForestRegressor(
                n_estimators=300, 
                max_depth=12, 
                min_samples_leaf=5, 
                random_state=42, 
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            test_preds = model.predict(X_test)
            model_name = "Random Forest"

        # Evaluate Tabular
        results_df, final_mse, final_mae = evaluate_sklearn_predictions(
            analyser=analyser,
            preds=test_preds,
            ids=data["id_test"],
            times=data["time_test"],
            model_name=model_name,
            class_mapping=pipeline.class_mapping
        )

    # ==========================================
    # ROUTE B: TIME-SERIES (PyTorch GRU)
    # ==========================================
    else:
        # Extract PyTorch DataLoaders
        train_loader, val_loader, test_loader = pipeline.get_dataloaders()
        
        sample_id, sample_X, sample_y, sample_time = next(iter(train_loader))
        print(f"Model: GRU (Detected {sample_X.shape[-1]} input features)")

        model = pipeline.build_model(hidden_dim=HIDDEN_DIM, embed_dim=embed_dim, dropout_rate=DROP_RATE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # Route Loss Function
        criterion = torch.nn.CrossEntropyLoss() if classification else torch.nn.MSELoss()
        task_type = 'classification' if classification else 'regression'

        trainer = Trainer(model, optimizer, criterion, task_type=task_type)
        trainer.fit(train_loader, val_loader=val_loader, num_epochs=EPOCHS, save_history=save_plotting)
        
        # Evaluate PyTorch
        results_df, final_mse, final_mae = evaluate_predictions(
            analyser=analyser,
            model=trainer.model, 
            dataloader=test_loader, 
            device=trainer.device,
            class_mapping=pipeline.class_mapping
        )

    # 3. Final Outputs
    # plot_prediction_distributions(results_df, resolution=UNIT, save=save_plotting)
    return results_df, final_mse, final_mae


def main():
    analyser = prepare_data()
    walk_forward_train(analyser, tabular=True)
    # train_classification_model(analyser, save_plotting=True)
    # train_random_forest_regression(analyser, save_plotting=True)
    # train_regression_model(analyser, save_plotting=True)

if __name__ == "__main__":
    main()
