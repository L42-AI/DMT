import torch

class BaseMLModel(torch.nn.Module):
    """ Base class that handles User ID Embedding logic for all model variants. """
    def __init__(self, num_users: int, embed_dim: int):
        super().__init__()
        # Lookup table for user embeddings. 
        self.user_embedding = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)

    def inject_embeddings(self, ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """ Looks up ID embeddings and concatenates them to the feature tensor. """
        embeds = self.user_embedding(ids) # Shape: [Batch, Embed_Dim]
        
        # If dealing with 3D Time-Series data [Batch, Seq_Len, Features]
        if x.dim() == 3:
            # Expand embeddings across the sequence length: [Batch, Seq_Len, Embed_Dim]
            embeds = embeds.unsqueeze(1).expand(-1, x.size(1), -1)
            
        # Concatenate along the final feature dimension
        return torch.cat([embeds, x], dim=-1)

class RandomClassificationBaseline(BaseMLModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_users: int, embed_dim: int = 5, dropout_rate: float = 0.5):
        super().__init__(num_users, embed_dim)
        self.output_dim = output_dim

    def forward(self, ids, x):
        """ Returns random probabilities for each class, securely attached to the graph. """
        batch_size = x.size(0)
        
        # 1. Generate the random scores (ensure it's on the correct device)
        rand_scores = torch.rand(batch_size, self.output_dim, device=x.device)
        
        # 2. Extract embeddings to get access to the model's trainable parameters
        embeds = self.user_embedding(ids)
        
        # 3. Stitch the graph together by adding 0 * the sum of the embeddings
        # This adds exactly 0.0 to the random scores, but forces PyTorch to track it backward!
        return rand_scores + (0.0 * embeds.sum())

class RandomRegressionBaseline(BaseMLModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_users: int, embed_dim: int = 5, dropout_rate: float = 0.5):
        super().__init__(num_users, embed_dim)
        self.output_dim = output_dim

    def forward(self, ids, x):
        """ Returns random probabilities for each class, securely attached to the graph. """
        batch_size = x.size(0)
        
        # 1. Generate the random scores (ensure it's on the correct device)
        rand_scores = torch.rand(batch_size, self.output_dim, device=x.device) * 9 + 1 
        
        # 2. Extract embeddings to get access to the model's trainable parameters
        embeds = self.user_embedding(ids)
        
        # 3. Stitch the graph together by adding 0 * the sum of the embeddings
        # This adds exactly 0.0 to the random scores, but forces PyTorch to track it backward!
        return rand_scores + (0.0 * embeds.sum())

class SimpleMLP(BaseMLModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_users: int, embed_dim: int = 5, dropout_rate: float = 0.5):
        super().__init__(num_users, embed_dim)
        # Note: The input layer now accommodates the original features + the embedding vector
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim + embed_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, ids, x):
        x_combined = self.inject_embeddings(ids, x)
        return self.net(x_combined)

class SimpleGRU(BaseMLModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_users: int, embed_dim: int = 5, dropout_rate: float = 0.5):
        super().__init__(num_users, embed_dim)
        self.gru = torch.nn.GRU(input_dim + embed_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, ids, x):
        x_combined = self.inject_embeddings(ids, x)
        out, _ = self.gru(x_combined)
        last_step_out = out[:, -1, :] 
        last_step_out = self.dropout(last_step_out)
        return self.fc(last_step_out)