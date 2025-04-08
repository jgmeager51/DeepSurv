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
    - Features columns (any number of them)
    - 'time' column: time to event/censoring
    - 'event' column: event indicator (1 = event occurred, 0 = censored)
    
    Returns:
        x: feature matrix
        t: time vector
        e: event indicator vector
    """
    # file_path = 'example_data.csv'
    df = pd.read_csv(file_path)
    
    # Extract time and event columns
    t = df['time'].values
    e = df['event'].values
    
    # Remove time and event columns from features
    x_df = df.drop(['time', 'event'], axis=1)
    
    # Convert to numpy array
    x = x_df.values
    
    return x, t, e

def main():
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # 1. Load data from CSV file
    print("Loading data...")
    try:
        x, t, e = load_data_from_csv('example_data.csv')
    except FileNotFoundError:
        # If file doesn't exist, create synthetic data for demonstration
        print("CSV file not found. Creating synthetic data for demonstration...")
        n_samples = 1000
        n_features = 10
        
        # Generate synthetic features
        x = np.random.randn(n_samples, n_features)
        
        # Generate synthetic survival times (exponential distribution with risk based on first 3 features)
        true_beta = np.zeros(n_features)
        true_beta[:3] = [0.5, -0.5, 0.2]  # Only first 3 features affect survival
        risk = np.dot(x, true_beta)
        scale = np.exp(risk)
        t = np.random.exponential(scale=np.exp(-risk))
        
        # Generate censoring (30% censoring rate)
        c = np.random.exponential(scale=np.exp(-risk) * 2)  # Censoring times
        e = (t <= c).astype(int)  # Event indicators
        t = np.minimum(t, c)  # Observed times
    
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
        learning_rate=0.01,
        hidden_layers_sizes=[32, 16],  # Two hidden layers with 32 and 16 neurons
        lr_decay=0.001,
        momentum=0.9,
        L2_reg=0.001,
        activation="relu",
        dropout=0.2,
        batch_norm=True,
        standardize=True  # Use built-in standardization
    )
    
    # Create a logger
    logger = DeepSurvLogger('DeepSurv_Example')
    
    # Train the model
    print("Training model...")
    metrics = model.train(
        train_data=train_data,
        valid_data=test_data,
        n_epochs=250,
        batch_size=64,  # Use batches of 64 samples
        validation_frequency=10,
        patience=50,  # Early stopping patience
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
    plt.savefig('deepsurv_training_history.png')
    print("Training history saved to 'deepsurv_training_history.png'")
    
    # 7. Save the trained model
    print("Saving model...")
    model.save_model('deepsurv_model.json', weights_file='deepsurv_weights.pt')
    print("Model saved as 'deepsurv_model.json' and 'deepsurv_weights.pt'")
    
    # 8. Feature importance visualization
    if x.shape[1] <= 10:  # Only for a reasonable number of features
        print("Calculating feature importance...")
        # Create a baseline risk
        baseline_risk = model.predict_risk(test_data['x'])
        
        # Calculate change in risk when each feature is perturbed
        feature_importance = []
        for i in range(x.shape[1]):
            x_perturbed = test_data['x'].copy()
            x_perturbed[:, i] += np.std(x_perturbed[:, i])  # Add one standard deviation
            perturbed_risk = model.predict_risk(x_perturbed)
            importance = np.mean(np.abs(perturbed_risk - baseline_risk))
            feature_importance.append(importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        feature_names = [f"Feature {i+1}" for i in range(x.shape[1])]
        plt.bar(feature_names, feature_importance)
        plt.xticks(rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance (Mean Absolute Risk Change)')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("Feature importance plot saved to 'feature_importance.png'")
    
    # 9. Risk surface visualization for first two features
    if x.shape[1] >= 2:
        print("Creating risk surface plot for first two features...")
        fig = model.plot_risk_surface(test_data['x'], i=0, j=1, figsize=(8, 6))
        plt.title('Risk Surface for Features 1 and 2')
        plt.savefig('risk_surface.png')
        print("Risk surface plot saved to 'risk_surface.png'")
    
    # 10. Demonstration of treatment recommendation
    if x.shape[1] >= 3:  # Using the third feature as treatment indicator
        print("\nDemonstrating treatment recommendation...")
        # Assume the third feature is a treatment indicator (0 or 1)
        treatment_idx = 2
        
        # Get treatment recommendations
        rec = model.recommend_treatment(
            x=test_data['x'],
            trt_i=1,  # Treatment 1
            trt_j=0,  # Treatment 0 (control)
            trt_idx=treatment_idx
        )
        
        # Print summary statistics
        print(f"Mean treatment recommendation: {np.mean(rec):.4f}")
        print(f"Percentage recommended for treatment: {np.mean(rec < 0) * 100:.1f}%")
        
        # Visualize distribution of treatment recommendations
        plt.figure(figsize=(8, 5))
        plt.hist(rec, bins=30)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Risk Difference (Treatment - Control)')
        plt.ylabel('Count')
        plt.title('Treatment Recommendation Distribution')
        plt.savefig('treatment_recommendation.png')
        print("Treatment recommendation plot saved to 'treatment_recommendation.png'")
    
    print("\nExecution complete!")

if __name__ == "__main__":
    main()