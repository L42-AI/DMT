from metric_accumulator import MetricAccumulator
import torch
from rich.console import Console
from rich.table import Table
import math
from sklearn.metrics import f1_score, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, model, optimizer, criterion, task_type='classification', device='cpu'):
        assert task_type in ['classification', 'regression'], "task_type must be 'classification' or 'regression'"
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.task_type = task_type
        self.device = device
        self.console = Console()
        
        # ADDED: Initialize a history dictionary to store metrics across epochs
        self.history = {'train': [], 'val': []}

    def _prepare_batch(self, batch):
        id_tensor, X_tensor, y_tensor, *_ = [b.to(self.device) for b in batch]
        return id_tensor, X_tensor, y_tensor

    def _compute_batch_metrics(self, loss, outputs, targets):
        metrics = {'loss': loss.item()}
        
        if self.task_type == 'classification':
            preds = outputs.argmax(dim=1)
            metrics['acc'] = (preds == targets).float().mean().item()
        else:
            metrics['mae'] = torch.abs(outputs - targets).mean().item()
            metrics['mse'] = torch.pow(outputs - targets, 2).mean().item()
            
        return metrics
    
    def _compute_epoch_metrics(self, all_preds, all_targets):
        if self.task_type == 'classification':
            return {
                'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
                'kappa': cohen_kappa_score(all_targets, all_preds, weights='quadratic')
            }
        return {}

    def _finalize_metrics(self, accumulator: MetricAccumulator, all_preds, all_targets):
        stats = accumulator.get_averages()
        
        if self.task_type == 'regression' and 'mse' in stats:
            stats['rmse'] = math.sqrt(stats['mse'])
            
        if all_preds and all_targets:
            extra = self._compute_epoch_metrics(all_preds, all_targets)
            stats.update(extra)
            
        return stats

    def _run_epoch(self, dataloader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        metrics = MetricAccumulator()
        
        all_preds, all_targets = [], []
        context = torch.set_grad_enabled(is_train)
        
        with context:
            for batch in dataloader:
                ids, X, y = self._prepare_batch(batch)
                
                if is_train: self.optimizer.zero_grad()
                
                outputs = self.model(ids, X)
                loss = self.criterion(outputs, y)
                
                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                batch_metrics = self._compute_batch_metrics(loss, outputs, y)
                metrics.update(batch_metrics)
                
                if self.task_type == 'classification':
                    all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                else:
                    all_preds.extend(outputs.detach().cpu().numpy().flatten())
                all_targets.extend(y.cpu().numpy().flatten())
                
        return self._finalize_metrics(metrics, all_preds, all_targets)

    def _print_header(self):
        self.console.rule(f"[bold]{self.task_type.title()} Pipeline Log[/bold]")
        table = Table(box=None, padding=(0, 2), show_header=True, show_edge=False)
        
        table.add_column("Ep", justify="right", style="cyan")
        table.add_column("T-Loss", justify="right", style="green")
        
        if self.task_type == 'classification':
            table.add_column("T-Acc", justify="right", style="magenta")
            table.add_column("V-Acc", justify="right", style="blue")
            table.add_column("V-F1", justify="right", style="bold blue")
            table.add_column("V-Kappa", justify="right", style="bold magenta")
        else:
            table.add_column("T-RMSE", justify="right", style="magenta")
            table.add_column("V-RMSE", justify="right", style="blue")
            table.add_column("V-MAE", justify="right", style="bold cyan")
            
        self.console.print(table)

    def _log_row(self, epoch, t_m, v_m=None):
        if not v_m: return 
        
        row = [f"{epoch:03d}", f"{t_m.get('loss', 0.0):.4f}"]
        
        if self.task_type == 'classification':
            row.extend([
                f"{t_m.get('acc', 0.0):.2%}", 
                f"{v_m.get('acc', 0.0):.2%}", 
                f"{v_m.get('f1_macro', 0.0):.3f}", 
                f"{v_m.get('kappa', 0.0):.3f}"
            ])
        else:
            row.extend([
                f"{t_m.get('rmse', 0.0):.4f}", 
                f"{v_m.get('rmse', 0.0):.4f}", 
                f"{v_m.get('mae', 0.0):.4f}"
            ])
            
        self.console.print("    ".join(row))

    def plot_training_curves(self, save_history=False):
        """ Generates learning curves with an in-plot magnification (inset) of the second half. """
        if not self.history['train']:
            print("No training history to plot.")
            return

        total_epochs = len(self.history['train'])
        epochs = list(range(1, total_epochs + 1))
        
        # We need at least 4 epochs to make a zoom-in mathematically meaningful
        show_zoom = total_epochs >= 4
        zoom_idx = total_epochs // 2 if show_zoom else 0
        
        sns.set_theme(style="whitegrid")
        # Back to 1 row, saving vertical space!
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), squeeze=False)
        fig.suptitle(f'{self.task_type.title()} Training History', fontsize=16, fontweight='bold', y=1.05)

        # --- 1. Data Extraction Helper ---
        def get_metric(split, key):
            return [epoch_data.get(key, 0) for epoch_data in self.history[split]]

        # --- 2. Inset Plotting Helper ---
        def plot_with_inset(ax, data_dict, title, is_loss=False):
            """ 
            Plots the main data and automatically generates an inset zoom.
            data_dict format: {'Label': (y_data, hex_color, line_style)}
            """
            # A. Plot Main Lines
            for label, (y_data, color, ls) in data_dict.items():
                ax.plot(epochs, y_data, label=label, color=color, lw=2, linestyle=ls)
            
            ax.set_title(title)
            ax.set_xlabel('Epochs')
            ax.legend(loc='best')
            
            # B. Plot Inset (Magnification)
            if show_zoom:
                # If it's a loss curve (goes down), put inset in top-right.
                # If it's an accuracy curve (goes up), put inset in bottom-right.
                # Format: [x0, y0, width, height] as fractions of the parent axis
                bounds = [0.45, 0.50, 0.5, 0.45] if is_loss else [0.45, 0.05, 0.5, 0.45]
                axins = ax.inset_axes(bounds)
                
                y_min, y_max = float('inf'), float('-inf')
                
                for label, (y_data, color, ls) in data_dict.items():
                    # Re-plot identical lines on the smaller inset axis
                    axins.plot(epochs, y_data, color=color, lw=2, linestyle=ls)
                    
                    # Track min/max of the zoomed region to set tight boundaries
                    zoom_y = y_data[zoom_idx:]
                    if zoom_y:
                        y_min = min(y_min, min(zoom_y))
                        y_max = max(y_max, max(zoom_y))
                
                # Add a 10% padding margin to the inset Y-axis
                margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 0.1
                
                # Lock the inset to the second half of epochs
                axins.set_xlim(epochs[zoom_idx], epochs[-1])
                axins.set_ylim(y_min - margin, y_max + margin)
                axins.tick_params(labelsize=8) # Smaller text for the inset
                
                # Draw the connecting lines from the box to the inset!
                ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.4)

        # --- 3. Gather Data and Execute ---
        ax_loss, ax_m1, ax_m2 = axes[0][0], axes[0][1], axes[0][2]
        
        # Subplot 1: Loss
        loss_data = {'Train Loss': (get_metric('train', 'loss'), '#1f77b4', '-')}
        if self.history['val']:
            loss_data['Val Loss'] = (get_metric('val', 'loss'), '#ff7f0e', '--')
        plot_with_inset(ax_loss, loss_data, 'Cross-Entropy / MSE Loss', is_loss=True)

        # Subplot 2 & 3: Task Specific Metrics
        if self.task_type == 'classification':
            m1_data = {'Train Acc': (get_metric('train', 'acc'), '#2ca02c', '-')}
            m2_data = {}
            if self.history['val']:
                m1_data['Val Acc'] = (get_metric('val', 'acc'), '#d62728', '--')
                m2_data['Val Macro-F1'] = (get_metric('val', 'f1_macro'), '#9467bd', '-')
                m2_data['Val Kappa'] = (get_metric('val', 'kappa'), '#8c564b', '-')
                
            plot_with_inset(ax_m1, m1_data, 'Accuracy')
            if m2_data: plot_with_inset(ax_m2, m2_data, 'Advanced Metrics (Validation)')
            
        else: # Regression
            m1_data = {'Train RMSE': (get_metric('train', 'rmse'), '#2ca02c', '-')}
            m2_data = {'Train MAE': (get_metric('train', 'mae'), '#9467bd', '-')}
            if self.history['val']:
                m1_data['Val RMSE'] = (get_metric('val', 'rmse'), '#d62728', '--')
                m2_data['Val MAE'] = (get_metric('val', 'mae'), '#8c564b', '--')
                
            plot_with_inset(ax_m1, m1_data, 'Root Mean Squared Error (RMSE)', is_loss=True)
            plot_with_inset(ax_m2, m2_data, 'Mean Absolute Error (MAE)', is_loss=True)

        plt.tight_layout()

        if save_history:
            plt.savefig(f'plots/{self.task_type}_training_history.png', bbox_inches='tight', dpi=300)
        else:
            plt.show()

    def fit(self, train_loader, val_loader=None, num_epochs=10, save_history=False):
        """ 
        The main entry point for the training loop. 
        
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation/test data.
            num_epochs: Number of epochs to train.
            save_history: If True, saves the training history.
        """
        self._print_header()
        
        # Reset history on new fit
        self.history = {'train': [], 'val': []}

        # --- 1. CORE TRAINING LOOP ---
        for epoch in range(1, num_epochs + 1):
            train_metrics = self._run_epoch(train_loader, is_train=True)
            self.history['train'].append(train_metrics)
            
            val_metrics = None
            if val_loader:
                val_metrics = self._run_epoch(val_loader, is_train=False)
                self.history['val'].append(val_metrics)
                
            self._log_row(epoch, train_metrics, val_metrics)
            
        # --- 2. END OF TRAINING VISUALIZATIONS ---
        print("\n" + "─" * 55)
        print("Training Complete. Generating visuals...")
        
        # Draw the standard Learning Curves (Loss, RMSE, etc.)
        self.plot_training_curves(save_history)