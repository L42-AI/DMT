import torch

from trainer import Trainer
from pipeline import TimeSeriesClassification

def train(analyser):
    # 1. Setup Data
    pipeline = TimeSeriesClassification(analyser, seq_len=7, num_bins=5)
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    # Safely extract the number of features from a single batch
    sample_id, sample_X, sample_y = next(iter(train_loader))
    num_features = sample_X.shape[2] # Shape is [batch, seq_len, num_features]

    # 2. Define a Time-Series Model
    class SimpleGRU(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, num_classes):
            super().__init__()
            # batch_first=True tells PyTorch our data is [batch, seq, features]
            self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = torch.nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            # Pass through GRU
            out, _ = self.gru(x)
            # Extract the hidden state from the very last time step in the sequence
            last_step_out = out[:, -1, :] 
            # Pass the final state to the classifier
            return self.fc(last_step_out)

    model = SimpleGRU(input_dim=num_features, hidden_dim=64, num_classes=pipeline.num_classes)

    # 3. Define Optimizer and Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 4. Train (Ensure you use the UniversalTrainer and pass the task_type)
    trainer = Trainer(model, optimizer, criterion, task_type='classification')
    trainer.fit(train_loader, val_loader=val_loader, num_epochs=200)