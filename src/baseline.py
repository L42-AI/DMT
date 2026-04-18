import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from analyser import Analyser
import data as _data
import multiprocessing as mp
from pipeline import TimeSeriesClassification
from functools import partial

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

def run_simulation_batch(worker_args, model_state, dataloader_args, device='cpu'):
    """
    Worker function executed by each CPU core.
    worker_args is now a tuple of (worker_id, num_sims_in_batch)
    """
    worker_id, num_sims_in_batch = worker_args
    
    import torch
    from torch.utils.data import DataLoader
    from pipeline import TimeSeriesClassification 
    
    model = model_state['class'](**model_state['args']).to(device)
    model.load_state_dict(model_state['weights'])
    model.eval()

    dataset = dataloader_args['dataset']
    loader = DataLoader(dataset, batch_size=dataloader_args['batch_size'], shuffle=False)

    batch_accuracies = []
    
    with torch.no_grad():
        # PROGRESS BAR MAGIC: Each core gets a unique position on the console
        for _ in tqdm(range(num_sims_in_batch), 
                      desc=f"Core {worker_id:02d}", 
                      position=worker_id, 
                      leave=False):
            
            correct = 0
            total = 0
            for batch in loader:
                ids, X, y = [b.to(device) for b in batch]
                outputs = model(ids, X)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                
            batch_accuracies.append(correct / total if total > 0 else 0)
            
    return batch_accuracies

def parallel_visualize_baseline(analyser, total_runs=100_000):
    num_cores = mp.cpu_count()
    print(f"\n--- Initializing Multiprocessed Visualizer ({num_cores} cores) ---")
    
    # 1. Prepare Data and Model Info
    pipeline = TimeSeriesClassification(analyser, seq_len=200, num_bins=5, batch_size=128)
    _, val_loader, _ = pipeline.get_dataloaders()
    model = pipeline.build_baseline_model()
    
    model_state = {
        'class': type(model),
        'args': {'output_dim': pipeline.num_classes, 
                 'num_users': pipeline.num_ids},
        'weights': model.state_dict()
    }
    
    loader_args = {
        'dataset': val_loader.dataset,
        'batch_size': val_loader.batch_size
    }

    # 2. Divide work into chunks WITH worker IDs
    runs_per_core = total_runs // num_cores
    worker_tasks = [(i, runs_per_core) for i in range(num_cores)]
    
    # Add the remainder to the last core
    worker_tasks[-1] = (num_cores - 1, runs_per_core + (total_runs % num_cores))

    # 3. Execution Pool
    print(f"Launching simulation...\n")
    accuracies = []
    
    # Setup context and a lock so progress bars don't overwrite each other
    ctx = mp.get_context('spawn')
    tqdm.set_lock(ctx.RLock())
    
    # We use initargs to pass the lock to all workers
    with ctx.Pool(processes=num_cores, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        
        worker_func = partial(run_simulation_batch, model_state=model_state, 
                              dataloader_args=loader_args, device='cpu')
        
        # We don't wrap the pool in tqdm anymore, because the workers are handling their own bars
        results = pool.map(worker_func, worker_tasks)
        
        # Flatten results list
        accuracies = [acc for sublist in results for acc in sublist]

    # Print a few newlines to clear the console area where the progress bars were
    print("\n" * num_cores)

    # 4. Plotting
    _plot_results(accuracies, total_runs)

def _plot_results(accuracies, total_runs):
    mean_acc = np.mean(accuracies)
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=50, color='mediumpurple', edgecolor='black')
    plt.axvline(mean_acc, color='red', linestyle='--', label=f'Baseline Mean: {mean_acc:.4f}')
    plt.title(f"Multiprocessed Baseline Distribution ({total_runs:,} runs)")
    plt.xlabel("Accuracy")
    plt.legend()
    plt.savefig("multiprocessed_baseline.png")
    print(f"Done. Average Baseline Accuracy: {mean_acc:.4%}")

if __name__ == "__main__":
    analyser = prepare_data()
    parallel_visualize_baseline(analyser, total_runs=10_000)