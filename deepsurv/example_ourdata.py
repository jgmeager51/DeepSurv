import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import the PyTorch DeepSurv implementation
from deepsurv_pytorch import DeepSurv, DeepSurvLogger

def load_data_from_csv(file_path):
    """
    Load survival data from a CSV file.

    Expected CSV format:
    - 'X1' and 'Y' columns: features for the risk function
    - 'time' column: time to event/censoring
    - 'event' column: event indicator (1 = event occurred, 0 = censored)

    Returns:
        x: feature matrix
        t: time vector
        e: event indicator vector
    """
    df = pd.read_csv(file_path)

    # Extract features, time, and event columns
    x = df[['X1', 'Y']].values  # Use 'X1' and 'Y' as features
    t = df['time'].values       # Time to event/censoring
    e = df['event'].astype(int).values  # Event indicator (1 = event, 0 = censored)

    return x, t, e

def main():
    # Set random seed for reproducibility
    np.random.seed(123)

    # 1. Load data from CSV file
    print("Loading data...")
    file_path = 'r_data2.csv'  # Path to the dataset
    x, t, e = load_data_from_csv(file_path)

    # Standardize the features
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 2. Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    x_train, x_test, t_train, t_test, e_train, e_test = train_test_split(
        x, t, e, test_size=0.2, random_state=42
    )

    # 3. Create data dictionaries
    train_data = {'x': x_train, 't': t_train, 'e': e_train}
    test_data = {'x': x_test, 't': t_test, 'e': e_test}

    # 4. Create and train DeepSurv model
    print("Creating DeepSurv model...")
    n_in = x.shape[1]  # Number of input features

    # Model hyperparameters
    model = DeepSurv(
        n_in=n_in,
        learning_rate=0.001,
        hidden_layers_sizes=[25, 25],  # Two hidden layers
        lr_decay=0.001,
        momentum=0.9,
        L2_reg=0.001,
        activation="relu",
        dropout=0.4,  # 20% dropout
        batch_norm=True,
        standardize=False  # Features are already standardized
    )

    # Create a logger
    logger = DeepSurvLogger('DeepSurv_R_Data2')

    # Train the model
    print("Training model...")
    metrics = model.train(
        train_data=train_data,
        valid_data=test_data,
        n_epochs=100,
        batch_size=32,
        validation_frequency=10,
        patience=20,
        logger=logger,
        verbose=True
    )

    # 5. Evaluate the model
    print("\nEvaluating model...")
    test_ci = model.get_concordance_index(
        x=test_data['x'],
        t=test_data['t'],
        e=test_data['e']
    )
    print(f"Test Concordance Index: {test_ci:.4f}")

    # 6. Plot training history
    print("Plotting training history...")
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['loss'], label='Training Loss')
    valid_epochs = [i for i, l in enumerate(metrics['valid_loss']) if l is not None]
    valid_loss = [l for l in metrics['valid_loss'] if l is not None]
    plt.plot(valid_epochs, valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot concordance index
    plt.subplot(1, 2, 2)
    plt.plot(metrics['c-index'], label='Training C-index')
    valid_cindex = [c for c in metrics['valid_c-index'] if c is not None]
    plt.plot(valid_epochs, valid_cindex, label='Validation C-index')
    plt.xlabel('Epoch')
    plt.ylabel('Concordance Index')
    plt.title('Training and Validation C-index')
    plt.legend()

    plt.tight_layout()
    plt.savefig('deepsurv_training_history_r_data2.png')
    print("Training history saved to 'deepsurv_training_history_r_data2.png'")

    # 7. Save the trained model
    print("Saving model...")
    model.save_model('deepsurv_model_r_data2.json', weights_file='deepsurv_weights_r_data2.pt')
    print("Model saved as 'deepsurv_model_r_data2.json' and 'deepsurv_weights_r_data2.pt'")

    # 8. Plot survival curves for the first five test samples
    print("Plotting survival curves for the first five test samples...")
    model.compute_baseline_hazard(x_train, t_train, e_train)  # Compute baseline hazard
    times = np.linspace(0, np.max(t_test), 100)  # Time grid for survival probabilities
    survival_probs = model.predict_survival(x_test[:5], times=times)  # Predict survival probabilities

    plt.figure(figsize=(10, 6))
    for i in range(5):
        plt.plot(times, survival_probs[i], label=f"Sample {i+1}")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.title("Predicted Survival Curves for First 5 Test Samples")
    plt.legend()
    plt.tight_layout()
    plt.savefig("survival_curves_r_data2.png")
    print("Survival curves plot saved to 'survival_curves_r_data2.png'")

    print("\nExecution complete!")

if __name__ == "__main__":
    main()