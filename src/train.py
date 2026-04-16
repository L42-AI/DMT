import sys
import torch

from pipeline import TimeSeriesClassification
from trainer import Trainer

from visualiser import Visualiser
from analyser import Analyser
import data as _data

def prepare_data() -> Analyser:
    print("--- Loading and Processing Raw Data ---")
    
    # Load the data
    data = _data.load()
    visualiser = Visualiser(data)
    analyser = Analyser(data)

    # Clean and aggregate
    analyser.process_outliers()
    analyser.aggregate.time_data(interval=1, unit='D', inplace=True)
    analyser.aggregate.activity(interval=1, unit='D', inplace=True)
    analyser.aggregate.communication_events(interval=1, unit='D', inplace=True)
    analyser.aggregate.reported_data(inplace=True)

    analyser.apply_scaling(inplace=True) # Scale before imputation to prevent data leakage from imputed values

    # Save intermediate state
    analyser.data.to_csv('data/aggregated_data.csv', index=False)
    
    # Impute missing values
    print("Running imputation...")
    analyser.impute(delete=False, catsi=True, epochs=10)
    analyser.data.to_csv('data/aggregated_data_after_impute.csv', index=False)

    # Quick diagnostic check
    missing_data = analyser.data.isna().any().sum()
    print(f"Columns with missing data after imputation: {missing_data}")
    
    return analyser

def train_model(analyser: Analyser):
    print("\n--- Initializing Machine Learning Pipeline ---")
    
    pipeline = TimeSeriesClassification(analyser, seq_len=7, num_bins=5)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    sample_id, sample_X, sample_y = next(iter(train_loader))
    batch, seq_len, num_features = sample_X.shape
    
    print(f"Detected {num_features} input features.")

    # 2. Instantiate the Model
    model = pipeline.create_model(input_dim=num_features, num_classes=pipeline.num_classes, hidden_dim=64, dropout_rate=0)

    # 3. Define Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, criterion, task_type='classification')
    trainer.fit(train_loader, val_loader=val_loader, num_epochs=200)

def main():
    analyser = prepare_data()
    train_model(analyser)

if __name__ == "__main__":
    main()
