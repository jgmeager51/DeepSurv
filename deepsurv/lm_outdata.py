import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from deepsurv_lm import LandmarkDeepSurv
import tracemalloc
import time

tracemalloc.start()


def load_r_data(file_path):
    """
    Load and preprocess the r_data.csv dataset.

    Parameters:
        file_path: Path to the r_data.csv file.

    Returns:
        Dictionary containing:
            - 'x': baseline covariates (n_samples, n_features)
            - 'z': longitudinal measurements (n_samples, n_timepoints, n_markers)
            - 't': event times (n_samples)
            - 'e': event indicators (n_samples)
    """
    df = pd.read_csv(file_path)

    # Extract fixed effect (X1), longitudinal response (Y), and measurement times (obstime)
    baseline_covariates = df[['X1']].drop_duplicates().values  # Fixed effect (X1)
    longitudinal_data = []
    for patient_id, group in df.groupby('id'):
        patient_measures = []
        for _, row in group.iterrows():
            patient_measures.append({
                'time': row['obstime'],
                'values': np.array([row['Y']])  # Longitudinal response (Y)
            })
        longitudinal_data.append(patient_measures)

    # Extract event times and event indicators
    event_times = df.groupby('id')['time'].first().values
    event_indicators = df.groupby('id')['event'].first().astype(int).values

    return {
        'x': baseline_covariates,  # Fixed effect (X1)
        'z': longitudinal_data,   # Longitudinal response (Y)
        't': event_times,         # Event times
        'e': event_indicators     # Event indicators
    }

def main():
    # Load the r_data.csv dataset
    file_path = 'r_data_50.csv'
    data = load_r_data(file_path)

    # Split into train/test (80/20)
    n_train = int(0.8 * len(data['x']))
    train_data = {
        'x': data['x'][:n_train],
        'z': data['z'][:n_train],
        't': data['t'][:n_train],
        'e': data['e'][:n_train]
    }
    test_data = {
        'x': data['x'][n_train:],
        'z': data['z'][n_train:],
        't': data['t'][n_train:],
        'e': data['e'][n_train:]
    }

    # Initialize and train the LandmarkDeepSurv model
    model = LandmarkDeepSurv(
        n_baseline_features=1,  # Number of baseline features (X1)
        n_longitudinal_features=1,  # Number of longitudinal markers (Y)
        landmark_times=[0.5, 1.0, 1.5, 2.0],  # Landmark times
        prediction_window=1.0,  # Prediction window
        hidden_layers_sizes=[32, 32],
        learning_rate=1e-3,
        dropout=0.4,
        batch_norm=True
    )
    start = time.time()
    # Train the model
    print("Training the model...")
    model.train(train_data, n_epochs=40, batch_size=32, validation_frequency=1,verbose=True)
    # Print training time
    print("Training time:")
    print(time.timer() - start)
    # for landmark_time, logger in model.loggers.items():
    #     print(f"Logs for Landmark Time {landmark_time}:")
    #     print(logger.history)

    # Evaluate the model on the test data
    print("\nEvaluating the model...")
    results = model.evaluate(test_data, metrics=['c-index', 'brier'])
    print("\nEvaluation Results:")
    for lm_time, metrics in results.items():
        print(f"Landmark {lm_time:.1f} years - C-index: {metrics['c-index']:.3f}, Brier: {metrics['brier']:.3f}")

    # Make dynamic predictions for the first test patient
    print("\nMaking dynamic predictions for the first test patient...")
    sample_idx = 0
    dynamic_preds, _ = model.predict_dynamic(
        test_data['x'][sample_idx:sample_idx+1],  # Baseline for one patient
        test_data['z'][sample_idx:sample_idx+1],  # Longitudinal for one patient
        times=[1.0, 2.0, 3.0]  # Times to predict at
    )
    print(f"\nDynamic predictions for patient {sample_idx}:")
    for t, pred in dynamic_preds.items():
        print(f"At t={t:.1f} years: Survival probability = {pred[0]:.2f}")

    __, df_dynamic_pred = model.predict_dynamic(test_data['x'], test_data['z'], times=[1.0, 2.0, 3.0])
    
    return model, results, df_dynamic_pred

import matplotlib.pyplot as plt

def plot_metrics(model, results):
    """
    Plot training/validation loss, c-index, and Brier score for each landmark time.

    Parameters:
        model: Trained LandmarkDeepSurv model
        results: Evaluation results containing c-index and Brier score
    """
    # Plot training and validation loss
    plt.figure(figsize=(12, 6))
    for landmark_time in model.landmark_times:
        plt.plot(model.training_loss[landmark_time], label=f"Train Loss (Landmark {landmark_time:.1f})")
        if landmark_time in model.validation_loss:
            plt.plot(model.validation_loss[landmark_time], label=f"Valid Loss (Landmark {landmark_time:.1f})", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss at Each Landmark Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_plot.png")
    print("Loss plot saved to 'loss_plot.png'")
    plt.show()

    # Plot c-index and Brier score
    c_indices = []
    brier_scores = []
    landmark_times = []

    for lm_time, metrics in results.items():
        landmark_times.append(lm_time)
        c_indices.append(metrics['c-index'])
        brier_scores.append(metrics['brier'])

    plt.figure(figsize=(12, 6))
    plt.plot(landmark_times, c_indices, marker='o', label="C-index")
    plt.plot(landmark_times, brier_scores, marker='o', label="Brier Score")
    plt.xlabel("Landmark Time (years)")
    plt.ylabel("Metric Value")
    plt.title("C-index and Brier Score at Each Landmark Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    print("Metrics plot saved to 'metrics_plot.png'")
    plt.show()



if __name__ == "__main__":
    model, results, df_pred = main()

    # Call this function after training and evaluation
    plot_metrics(model, results)

    predictions_df = pd.DataFrame(df_pred)
    predictions_df.to_csv("dynamic_predictions.csv", index=False)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

tracemalloc.stop()