import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def _get_predictions(model, dataloader, device, freq='D'):
    """Runs model inference and aggregates predictions to the specified timeframe."""
    model.eval()
    ids, times, preds = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            id_t, X_t, _, time_t = [b.to(device) for b in batch]
            ids.extend(id_t.cpu().numpy())
            times.extend(time_t.cpu().numpy())
            preds.extend(model(id_t, X_t).cpu().numpy().flatten())
            
    df = pd.DataFrame({'id': ids, 'timestamp': times, 'predicted_mood': preds})
    
    # Format time and rescale (0-1 -> 1-10)
    df['period'] = pd.to_datetime(df['timestamp'], unit='s').dt.floor(freq)
    df['predicted_mood'] = df['predicted_mood'] * 9 + 1
    
    return df.groupby(['id', 'period'])['predicted_mood'].mean().reset_index()

def _get_ground_truth(analyser, freq='D'):
    """Extracts, formats, and aggregates the true mood values from the Analyser."""
    df = analyser.data[analyser.data['variable'] == 'mood'].copy()
    
    # Format ID, floor time, and rescale
    df['id'] = df['id'].apply(lambda x: int(str(x)[-2:])).astype('category').cat.codes
    df['period'] = pd.to_datetime(df['time'], unit='s').dt.floor(freq)
    df['actual_mood'] = df['value'] * 9 + 1
    
    return df.groupby(['id', 'period'])['actual_mood'].mean().reset_index()

def evaluate_predictions(analyser, model, dataloader, device='cpu'):
    """
    Evaluates model predictions against the absolute ground truth.
    
    Args:
        analyser: Data analyser instance containing the raw dataset.
        model: Trained PyTorch model.
        dataloader: DataLoader containing the evaluation dataset.
        device: Computing device ('cpu', 'cuda', etc.).
        
    Returns:
        results_df: DataFrame containing the merged actual and predicted values.
        mse: Mean Squared Error.
        mae: Mean Absolute Error.
    """
    freq = 'D'
    
    pred_agg = _get_predictions(model, dataloader, device, freq)
    truth_agg = _get_ground_truth(analyser, freq)
    
    # Inner merge implicitly filters out non-overlapping periods
    results_df = pd.merge(pred_agg, truth_agg, on=['id', 'period'], how='inner')
    
    mse = mean_squared_error(results_df['actual_mood'], results_df['predicted_mood'])
    mae = mean_absolute_error(results_df['actual_mood'], results_df['predicted_mood'])
    
    print(f"\n{'='*45}\n🎯 SOURCE-OF-TRUTH METRICS ({freq})\n{'='*45}")
    print(f"Successfully Merged Windows: {len(results_df):,}")
    print(f"Mean Squared Error (MSE):    {mse:.4f}")
    print(f"Mean Absolute Error (MAE):   {mae:.4f}\n{'='*45}")
    
    return results_df, mse, mae


def plot_prediction_distributions(results_df: pd.DataFrame, resolution='D', save=False):
    """
    Plots the Kernel Density Estimates of True vs. Predicted vs. Random mood.
    
    Args:
        results_df: Merged DataFrame containing 'actual_mood' and 'predicted_mood'.
        resolution: Aggregation timeframe (e.g., 'D', 'H') for the file naming.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))

    # Generate uniform random baseline
    random_baseline = np.random.uniform(1.0, 10.0, size=len(results_df))

    # Plot distributions
    sns.kdeplot(data=results_df['actual_mood'], fill=True, color="#1f77b4", alpha=0.4, label="True Mood")
    sns.kdeplot(data=results_df['predicted_mood'], fill=True, color="#2ca02c", alpha=0.4, label="Predicted Mood")
    sns.kdeplot(data=random_baseline, fill=False, color="#d62728", ls="--", label="Random Guess")

    # Add mean lines
    plt.axvline(results_df['actual_mood'].mean(), color='#1f77b4', ls=':', alpha=0.8)
    plt.axvline(results_df['predicted_mood'].mean(), color='#2ca02c', ls=':', alpha=0.8)

    # Formatting
    plt.title(f"{resolution.title()} Average Mood: True vs. Predicted vs. Random", fontsize=16, fontweight='bold')
    plt.xlabel("Mood Score (1 - 10)")
    plt.ylabel("Density")
    plt.xlim(1, 10) 
    plt.legend()
    plt.tight_layout()
    
    # Save output
    if save:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{resolution.lower()}_mood_distributions.png", dpi=300, bbox_inches='tight')
        plt.gca()
    else:
        plt.show()
