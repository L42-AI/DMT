import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

# Make sure these match your actual file structure
from analyser import Analyser
import data as _data
from pipeline import TimeSeriesClassification, TimeSeriesRegression

# --- HYPERPARAMETERS ---
TOTAL_RUNS = 100_000 # Lowered to 10k so testing 4 different classes doesn't take all day!
TEST_CLASSES = [2, 3, 4, 5] 

def prepare_data() -> Analyser:
    print("--- Loading and Processing Raw Data ---")
    data = _data.load()
    analyser = Analyser(data)

    analyser.process_outliers()
    analyser.aggregate.time_data(interval=1, unit='H', inplace=True)
    analyser.aggregate.activity(interval=1, unit='H', inplace=True)
    analyser.aggregate.communication_events(interval=1, unit='H', inplace=True)
    analyser.aggregate.reported_data(interval=1, unit='H', inplace=True)

    analyser.apply_scaling(inplace=True)

    missing_data = analyser.data.isna().any().sum()
    print(f"Columns with missing data after imputation: {missing_data}")
    
    return analyser

def run_simulation_batch(worker_args, model_state, dataloader_args, task_type, device='cpu'):
    """ Worker function executed by each CPU core. """
    worker_id, num_sims_in_batch = worker_args
    
    model = model_state['class'](**model_state['args']).to(device)
    model.load_state_dict(model_state['weights'])
    model.eval()

    dataset = dataloader_args['dataset']
    loader = DataLoader(dataset, batch_size=dataloader_args['batch_size'], shuffle=False)

    batch_results = []
    
    with torch.no_grad():
        for _ in tqdm(range(num_sims_in_batch), 
                      desc=f"Core {worker_id:02d}", 
                      position=worker_id, 
                      leave=False):
            
            accumulated_metric = 0.0
            total_items = 0
            
            for batch in loader:
                ids, X, y, *_ = [b.to(device) for b in batch] 
                outputs = model(ids, X)
                
                if task_type == 'classification':
                    predictions = outputs.argmax(dim=1)
                    accumulated_metric += (predictions == y).sum().item()
                else: 
                    accumulated_metric += F.mse_loss(outputs, y, reduction='sum').item()
                
                total_items += y.size(0)
                
            if total_items > 0:
                if task_type == 'classification':
                    final_metric = accumulated_metric / total_items
                else:
                    mse = accumulated_metric / total_items
                    final_metric = math.sqrt(mse) 
            else:
                final_metric = 0.0
                
            batch_results.append(final_metric)
            
    return batch_results

def parallel_visualize_baseline(analyser, task_type='classification', total_runs=10_000, num_classes=5):
    num_cores = mp.cpu_count()
    print(f"\n" + "="*50)
    print(f"🚀 INITIALIZING {task_type.upper()} SIMULATION (Classes: {num_classes if task_type == 'classification' else 'N/A'})")
    print(f"Cores: {num_cores} | Total Runs: {total_runs:,}")
    print("="*50)
    
    # Prepare Pipeline dynamically
    if task_type == 'classification':
        pipeline = TimeSeriesClassification(analyser, seq_len=24, num_classes=num_classes, batch_size=2048)
    else:
        pipeline = TimeSeriesRegression(analyser, seq_len=24, batch_size=2048)
        
    _, val_loader, _ = pipeline.get_dataloaders()
    model = pipeline.build_baseline_model()
    
    model_state = {
        'class': type(model),
        'args': {'output_dim': pipeline.num_classes, 
                 'num_ids': pipeline.num_ids},
        'weights': model.state_dict()
    }
    
    loader_args = {
        'dataset': val_loader.dataset,
        'batch_size': val_loader.batch_size
    }

    # Divide work into chunks
    runs_per_core = total_runs // num_cores
    worker_tasks = [(i, runs_per_core) for i in range(num_cores)]
    worker_tasks[-1] = (num_cores - 1, runs_per_core + (total_runs % num_cores))

    # Execution Pool
    print(f"Launching Multiprocessing Pool...\n")
    results = []
    
    ctx = mp.get_context('spawn')
    tqdm.set_lock(ctx.RLock())
    
    with ctx.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        worker_func = partial(run_simulation_batch, 
                              model_state=model_state, 
                              dataloader_args=loader_args, 
                              task_type=task_type,
                              device='cpu')
        
        raw_results = pool.map(worker_func, worker_tasks)
        results = [metric for sublist in raw_results for metric in sublist]

    print("\n" * num_cores)
    return results # Return results instead of plotting immediately

def plot_multi_class_grid(class_results_dict, total_runs):
    """ Plots a 2x2 grid to mathematically prove the baseline accuracy distributions. """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, (num_classes, results) in enumerate(class_results_dict.items()):
        ax = axes[idx]
        ax.set_facecolor('#F8F9FA')
        
        mean_val = np.mean(results)
        std_val = np.std(results)
        expected_math = 1.0 / num_classes
        
        ax.hist(results, bins=30, color='#4C72B0', edgecolor='black', alpha=0.85, linewidth=0.5)
        ax.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=2, label=f'Empirical Mean: {mean_val:.4f}')
        ax.axvline(expected_math, color='#2CA02C', linestyle=':', linewidth=2, label=f'Expected ($1/{num_classes}$): {expected_math:.4f}')
        ax.axvspan(mean_val - std_val, mean_val + std_val, color='#C44E52', alpha=0.15)

        ax.set_title(f"{num_classes} Classes", fontsize=14, fontweight='bold')
        ax.set_xlabel("Accuracy", fontsize=10, fontweight='bold')
        ax.set_ylabel("Frequency", fontsize=10, fontweight='bold')
        
        ax.grid(True, linestyle=':', alpha=0.7, color='grey', axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)

    plt.suptitle(f"Monte Carlo Classification Baseline across Class Counts\n({total_runs:,} Simulations per Grid)", fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    filename = "baseline_multiclass_proof.png"
    plt.savefig(filename, bbox_inches='tight', transparent=False)
    print(f"✅ Saved Multi-Class Proof to: '{filename}'")
    plt.close()

def plot_single_regression(results, total_runs):
    """ Generates the single RMSE plot for the regression baseline. """
    mean_val = np.mean(results)
    std_val = np.std(results)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    ax.hist(results, bins='auto', color='#DD8452', edgecolor='black', alpha=0.85, linewidth=0.5)
    ax.axvline(mean_val, color='#C44E52', linestyle='--', linewidth=2, label=f'Expected Baseline: {mean_val:.4f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='#C44E52', alpha=0.15, label=f'± 1 Std Dev ({std_val:.4f})')

    ax.set_title("Random Baseline Monte Carlo Simulation (Regression)\n", fontsize=16, fontweight='bold')
    ax.set_xlabel("Model Performance (RMSE)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold', labelpad=10)
    
    ax.grid(True, linestyle=':', alpha=0.7, color='grey', axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    stats_text = f"Simulations: {total_runs:,}\nMean: {mean_val:.4f}\nStd Dev: {std_val:.4f}"
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11, verticalalignment='top', bbox=props, family='monospace')

    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    filename = "baseline_distribution_regression.png"
    plt.savefig(filename, bbox_inches='tight', transparent=False)
    print(f"✅ Saved Regression plot to: '{filename}'")
    plt.close()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True) 
    
    analyser = prepare_data()
    
    # 1. Run the Multi-Class Classification Proof
    all_class_results = {}
    for c in TEST_CLASSES:
        results = parallel_visualize_baseline(analyser, task_type='classification', total_runs=TOTAL_RUNS, num_classes=c)
        all_class_results[c] = results
        
    plot_multi_class_grid(all_class_results, TOTAL_RUNS)
    
    # 2. Run the standalone Regression Simulation
    reg_results = parallel_visualize_baseline(analyser, task_type='regression', total_runs=TOTAL_RUNS)
    plot_single_regression(reg_results, TOTAL_RUNS)