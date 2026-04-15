from socket import gethostname
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_



def fit(model, train_iter, valid_iter, epochs=300, device=torch.device('cpu')):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_train = 0
        
        for batch in train_iter:
            optimizer.zero_grad()
            ret = model(batch)
            ret['loss'].backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            train_loss += ret['loss'].item() * ret['loss_count'].item()
            n_train += ret['loss_count'].item()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            n_val = 0
            for batch in valid_iter:
                ret = model(batch)
                val_loss += ret['loss'].item() * ret['loss_count'].item()
                n_val += ret['loss_count'].item()
        print(f'Epoch {epoch+1}: train_loss={train_loss/n_train:.3e}, val_loss={val_loss/n_val:.3e}')
