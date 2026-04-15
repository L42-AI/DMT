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

    def _calculate_metrics(self, outputs, targets):
        """ Dynamically calculates the right metric based on the task. """
        if self.task_type == 'classification':
            return {'acc': (outputs.argmax(dim=1) == targets).float().mean().item()}
        else: # regression
            return {'mae': torch.abs(outputs - targets).mean().item()}

    def _print_header(self):
        """ Dynamically builds the table header. """
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

    def fit(self, train_loader, val_loader=None, num_epochs=10):
        self._print_header()
        
        # Determine which metric key we are looking for when formatting the string
        m_key = 'acc' if self.task_type == 'classification' else 'mae'

        for epoch in range(1, num_epochs + 1):
            
            # --- 1. TRAINING PHASE ---
            self.model.train()
            train_metrics = MetricAccumulator()
            
            for batch in train_loader:
                id_tensor, X_tensor, y_tensor = [b.to(self.device) for b in batch]
                
                self.optimizer.zero_grad()
                outputs = self.model(X_tensor) 
                loss = self.criterion(outputs, y_tensor)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Calculate base loss + dynamic task metrics
                batch_metrics = {'loss': loss.item(), 'acc': (outputs.argmax(dim=1) == y_tensor).float().mean().item()} if self.task_type == 'classification' else {'loss': loss.item(), 'mae': torch.abs(outputs - y_tensor).mean().item()}
                batch_metrics.update(self._calculate_metrics(outputs, y_tensor))
                train_metrics.update(batch_metrics)

            avg_train = train_metrics.get_averages()

            # --- 2. VALIDATION PHASE ---
            val_loss_str = "---"
            val_metric_str = "---"
            
            if val_loader:
                self.model.eval() 
                val_metrics = MetricAccumulator()
                
                with torch.no_grad():
                    for batch in val_loader:
                        id_tensor, X_tensor, y_tensor = [b.to(self.device) for b in batch]
                        
                        outputs = self.model(X_tensor)
                        loss = self.criterion(outputs, y_tensor)
                        
                        batch_metrics = {'loss': loss.item(), 'acc': (outputs.argmax(dim=1) == y_tensor).float().mean().item()} if self.task_type == 'classification' else {'loss': loss.item(), 'mae': torch.abs(outputs - y_tensor).mean().item()}
                        batch_metrics.update(self._calculate_metrics(outputs, y_tensor))
                        val_metrics.update(batch_metrics)
                        
                avg_val = val_metrics.get_averages()
                val_loss_str = f"{avg_val['loss']:.4f}"
                
                # Format validation metric (percentage for Acc, float for MAE)
                if self.task_type == 'classification':
                    val_metric_str = f"{avg_val[m_key]:.1%}"
                else:
                    val_metric_str = f"{avg_val[m_key]:.4f}"

            # --- 3. LOGGING ---
            row_table = Table(box=None, padding=(0, 2), show_header=False, show_edge=False)
            
            row_table.add_column(justify="right", style="cyan")    # Epoch
            row_table.add_column(justify="right", style="green")   # Train Loss
            row_table.add_column(justify="right", style="magenta") # Train Metric
            row_table.add_column(justify="right", style="yellow")  # Val Loss
            row_table.add_column(justify="right", style="blue")    # Val Metric
            
            # Format train metric
            train_metric_val = avg_train.get(m_key, 0.0)
            if self.task_type == 'classification':
                train_metric_str = f"{train_metric_val:.1%}"
            else:
                train_metric_str = f"{train_metric_val:.4f}"
            
            row_table.add_row(
                f"{epoch:03d}", 
                f"{avg_train.get('loss', 0.0):.4f}", 
                train_metric_str, 
                val_loss_str,
                val_metric_str
            )
            self.console.print(row_table)