import optuna
import warnings

# Suppress warnings to keep the parallel output clean
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.INFO)

from train import prepare_data, train_model

def run_behavioral_tuning(analyser, classification: bool, tabular: bool, n_trials=30):
    """
    Runs a multi-core hyperparameter search focusing exclusively on 
    behavioral parameters, allowing for deep historical lookbacks.
    """
    
    def objective(trial):
        # 1. Tune the complexity of the "Personalities"
        # We know 3 performed well, so we keep the search tight around the lower end
        embed_dim = trial.suggest_categorical('embed_dim', [2, 3, 4, 5, 8, 10])
        
        # 2. Tune the "Emotional Memory" (Lookback Horizon IN HOURS)
        if tabular:
            # Expanded Tabular combinations
            # [Week, 2-Weeks], [Week, 3-Weeks], [2-Weeks, 4-Weeks], [Week, 2-Weeks, 4-Weeks]
            window_choices = [
                [24, 168],        # 1 Day, 1 Week (Baseline comparison)
                [168, 336],       # 1 Week, 2 Weeks
                [168, 504],       # 1 Week, 3 Weeks
                [336, 672],       # 2 Weeks, 4 Weeks
                [168, 336, 672]   # 1 Week, 2 Weeks, 4 Weeks (Dense historical features)
            ]
            window_idx = trial.suggest_categorical('window_idx', [0, 1, 2, 3, 4])
            windows = window_choices[window_idx]
            seq_len = 24 # Ignored by tabular
        else:
            # Expanded GRU Cycles
            # 72h (3 Days), 168h (1W), 336h (2W), 504h (3W), 672h (4W), 1008h (6W)
            seq_len = trial.suggest_categorical('seq_len', [72, 168, 336, 504, 672, 1008])
            windows = [24, 168] # Ignored by TimeSeries
            
        # 3. Run the pipeline
        try:
            # plot visalizing is strictly disabled here!
            _, final_mse, final_mae = train_model(
                analyser=analyser,
                classification=classification,
                tabular=tabular,
                seq_len=seq_len,
                embed_dim=embed_dim,
                windows=windows,
                save_plotting=False 
            )
        except Exception as e:
            # If a sequence length exceeds the available data for users, prune and move on
            print(f"⚠️  Trial failed (likely sequence length {seq_len} exceeded user data history). Error: {e}. Pruning this trial.")
            raise optuna.TrialPruned()

        # 4. We want to minimize the Mean Absolute Error (MAE)
        return final_mae

    # --- EXECUTE THE STUDY ---
    study_name = f"DeepTuning_{'Tabular' if tabular else 'TimeSeries'}_{'Class' if classification else 'Reg'}"
    study = optuna.create_study(direction='minimize', study_name=study_name)
    
    print("\n" + "="*50)
    print(f"🚀 STARTING MULTI-CORE OPTUNA STUDY: {study_name}")
    print("="*50)
    
    # n_jobs=-1 distributes the workload across ALL your CPU cores
    study.optimize(objective, n_trials=n_trials, n_jobs=-1)

    # --- RESULTS ---
    print("\n" + "="*50)
    print("🏆 OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best MAE: {study.best_value:.4f}")
    print("\nOptimal Behavioral Parameters:")
    
    # Map the window_idx back to the actual list for readability if tabular
    best_params = study.best_params
    if tabular and 'window_idx' in best_params:
        window_choices = [
            [24, 168], 
            [168, 336], 
            [168, 504], 
            [336, 672], 
            [168, 336, 672]
        ]
        best_params['windows'] = window_choices[best_params.pop('window_idx')]
        
    for key, value in best_params.items():
        print(f"  ➔ {key}: {value}")
    print("="*50)

    return study

def main():
    analyser = prepare_data()
    
    # Run the tuning loop
    study = run_behavioral_tuning(analyser, classification=True, tabular=False, n_trials=30)
    
    # Extract the winning variables
    best_seq = study.best_params['seq_len']
    best_embed = study.best_params['embed_dim']
    
    # Run one final time with plotting enabled to save your graphs
    print(f"\n📸 Running final model with optimal parameters (seq_len={best_seq}, embed_dim={best_embed})...")
    train_model(
        analyser, 
        classification=True, 
        tabular=True, 
        seq_len=best_seq, 
        embed_dim=best_embed, 
        save_plotting=True
    )

if __name__ == "__main__":
    main()