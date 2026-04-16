from metric_accumulator import MetricAccumulator
import torch
from rich.console import Console
from rich.table import Table

class Trainer:
    def __init__(self, model, optimizer, criterion, task_type='classification', device='cpu'):
        assert task_type in ['classification', 'regression'], "task_type must be 'classification' or 'regression'"
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_type = task_type
        self.device = device
        self.console = Console()
        
        # Set the metric key based on task type
        self.m_key = 'acc' if self.task_type == 'classification' else 'mae'

    def _compute_metrics(self, loss, outputs, targets):
        """ Single source of truth for metric calculations. """
        metrics = {'loss': loss.item()}
        
        if self.task_type == 'classification':
            preds = outputs.argmax(dim=1)
            metrics['acc'] = (preds == targets).float().mean().item()
        else: # regression
            metrics['mae'] = torch.abs(outputs - targets).mean().item()
            
        return metrics

    def _train_epoch(self, dataloader):
        """ Handles the training loop for a single epoch. """
        self.model.train()
        metrics = MetricAccumulator()
        
        for batch in dataloader:
            id_tensor, X_tensor, y_tensor, *_ = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            outputs = self.model(id_tensor, X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_metrics = self._compute_metrics(loss, outputs, y_tensor)
            metrics.update(batch_metrics)
            
        return metrics.get_averages()

    def _validate_epoch(self, dataloader):
        """ Handles the validation loop for a single epoch. """
        self.model.eval() 
        metrics = MetricAccumulator()
        
        with torch.no_grad():
            for batch in dataloader:
                id_tensor, X_tensor, y_tensor, *_ = [b.to(self.device) for b in batch]
                
                outputs = self.model(id_tensor, X_tensor)
                loss = self.criterion(outputs, y_tensor)
                
                batch_metrics = self._compute_metrics(loss, outputs, y_tensor)
                metrics.update(batch_metrics)
                
        return metrics.get_averages()

    def _print_header(self):
        """ Builds the table header dynamically. """
        self.console.rule(f"[bold]{self.task_type.title()} Training Log[/bold]")
        table = Table(box=None, padding=(0, 2), show_header=True, show_edge=False)
        
        metric_name = "Acc" if self.task_type == 'classification' else "MAE"
        
        table.add_column("Epoch", justify="right", style="cyan")
        table.add_column("Train Loss", justify="right", style="green")
        table.add_column(f"Train {metric_name}", justify="right", style="magenta")
        table.add_column("Val Loss", justify="right", style="yellow")
        table.add_column(f"Val {metric_name}", justify="right", style="blue")
        
        self.console.print(table)
        self.console.print("─" * 55, style="dim")

    def _log_epoch(self, epoch, train_metrics, val_metrics=None):
        """ Handles the rich table row formatting for the epoch. """
        row_table = Table(box=None, padding=(0, 2), show_header=False, show_edge=False)
        
        row_table.add_column(justify="right", style="cyan")    
        row_table.add_column(justify="right", style="green")   
        row_table.add_column(justify="right", style="magenta") 
        row_table.add_column(justify="right", style="yellow")  
        row_table.add_column(justify="right", style="blue")    
        
        # Format Train Metrics
        train_loss = f"{train_metrics.get('loss', 0.0):.4f}"
        train_val = train_metrics.get(self.m_key, 0.0)
        train_m_str = f"{train_val:.1%}" if self.task_type == 'classification' else f"{train_val:.4f}"

        # Format Validation Metrics
        val_loss = "---"
        val_m_str = "---"
        if val_metrics:
            val_loss = f"{val_metrics.get('loss', 0.0):.4f}"
            val_val = val_metrics.get(self.m_key, 0.0)
            val_m_str = f"{val_val:.1%}" if self.task_type == 'classification' else f"{val_val:.4f}"
        
        row_table.add_row(f"{epoch:03d}", train_loss, train_m_str, val_loss, val_m_str)
        self.console.print(row_table)

    def fit(self, train_loader, val_loader=None, num_epochs=10):
        """ The main entry point for the training loop. """
        self._print_header()

        for epoch in range(1, num_epochs + 1):
            train_metrics = self._train_epoch(train_loader)
            
            val_metrics = None
            if val_loader:
                val_metrics = self._validate_epoch(val_loader)
                
            self._log_epoch(epoch, train_metrics, val_metrics)