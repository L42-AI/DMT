import torch

class SimpleMLP(torch.nn.Module):
    """ Standard Feed-Forward Network for Tabular Data (2D) """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class SimpleGRU(torch.nn.Module):
    """ Recurrent Network for Time-Series Data (3D) """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float = 0.5):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        last_step_out = out[:, -1, :] 
        last_step_out = self.dropout(last_step_out)
        return self.fc(last_step_out)
