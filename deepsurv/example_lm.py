
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from deepsurv_lm import LandmarkDeepSurv

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for heart valve surgery patients
def generate_synthetic_data(n_patients=200, n_timepoints=5, max_followup=5):
    """
    Generate synthetic dataset mimicking heart valve surgery patients with:
    - Baseline covariates
    - Longitudinal measurements (lvmi, grad)
    - Survival times and event indicators
    """
    # Baseline characteristics (15 features as in the paper)
    baseline_data = {
        'age': np.random.normal(60, 10, n_patients),
        'sex': np.random.binomial(1, 0.6, n_patients),  # 60% male
        'bsa': np.random.normal(1.8, 0.2, n_patients),  # body surface area
        'emergenc': np.random.binomial(1, 0.1, n_patients),  # emergency surgery
        'hs': np.random.binomial(1, 0.5, n_patients),  # valve prosthesis type
        'dm': np.random.binomial(1, 0.2, n_patients),  # diabetes
        'hc': np.random.binomial(1, 0.3, n_patients),  # high cholesterol
        'prenyha': np.random.randint(1, 5, n_patients),  # NYHA class
        'lvh': np.random.binomial(1, 0.4, n_patients),  # left ventricular hypertrophy
        'creat': np.random.lognormal(1.1, 0.3, n_patients),  # creatinine
        'acei': np.random.binomial(1, 0.4, n_patients),  # ACE inhibitor use
        'con_cabg': np.random.binomial(1, 0.3, n_patients),  # concomitant CABG
        'sten_reg_mix': np.random.randint(0, 3, n_patients),  # valve hemodynamics
        'lv': np.random.normal(55, 10, n_patients),  # LV ejection fraction
        'size': np.random.choice([19, 21, 23, 25], n_patients)  # valve size
    }
    
    # Create baseline dataframe
    df_baseline = pd.DataFrame(baseline_data)
    
    # Standardize continuous variables
    cont_vars = ['age', 'bsa', 'creat', 'lv']
    scaler = StandardScaler()
    df_baseline[cont_vars] = scaler.fit_transform(df_baseline[cont_vars])
    
    # Generate longitudinal measurements (lvmi and grad)
    longitudinal_data = []
    for i in range(n_patients):
        patient_measures = []
        # Generate measurement times (irregularly spaced)
        measure_times = sorted(np.random.uniform(0, max_followup, n_timepoints))
        
        # Base values influenced by baseline characteristics
        base_lvmi = 100 + df_baseline.loc[i, 'age']*0.5 + df_baseline.loc[i, 'lvh']*20
        base_grad = 10 + df_baseline.loc[i, 'hs']*5 - df_baseline.loc[i, 'lv']*0.1
        
        for t in measure_times:
            # Add some random progression over time
            lvmi = base_lvmi + t*2 + np.random.normal(0, 5)
            grad = base_grad + t*0.5 + np.random.normal(0, 1)
            
            patient_measures.append({
                'time': t,
                'values': np.array([lvmi, grad])  # lvmi and grad measurements
            })
        longitudinal_data.append(patient_measures)
    
    # Generate survival times (Weibull distribution)
    scale = max_followup * 0.7
    shape = 1.5
    
    # Make survival times depend on covariates
    X = df_baseline.values
    coefs = np.random.randn(X.shape[1]) * 0.1
    linear_predictor = np.dot(X, coefs)
    
    # Generate event times (Weibull with covariate effects)
    event_times = scale * np.random.weibull(shape, n_patients) * np.exp(-linear_predictor/shape)
    
    # Generate censoring times (uniform over follow-up period)
    censoring_times = np.random.uniform(0.5*max_followup, max_followup, n_patients)
    
    # Observed time is min of event and censoring time
    observed_times = np.minimum(event_times, censoring_times)
    event_indicators = (event_times <= censoring_times).astype(int)
    
    # Convert baseline data to numpy array
    baseline_array = df_baseline.values
    
    return {
        'x': baseline_array,  # Baseline covariates (n_patients, 15)
        'z': longitudinal_data,  # Longitudinal measurements
        't': observed_times,  # Event/censoring times
        'e': event_indicators  # Event indicators (1=event, 0=censored)
    }

# Generate synthetic data
data = generate_synthetic_data(n_patients=200)

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

# Initialize and train the model
model = LandmarkDeepSurv(
    n_baseline_features=15,  # Number of baseline features
    n_longitudinal_features=2,  # lvmi and grad
    landmark_times=[0.5, 1.0, 1.5, 2.0, 2.5],  # Landmark times (years)
    prediction_window=1.0,  # 3-year prediction window
    hidden_layers_sizes=[32, 32],
    learning_rate=1e-3,
    dropout=0.5,
    batch_norm=True
)

# Train the model
model.train(train_data, n_epochs=100, verbose=True)

# Evaluate on test data
results = model.evaluate(test_data, metrics=['c-index', 'brier'])
print("\nEvaluation Results:")
for lm_time, metrics in results.items():
    print(f"Landmark {lm_time:.1f} years - C-index: {metrics['c-index']:.3f}, Brier: {metrics['brier']:.3f}")

# Make dynamic predictions for first test patient
sample_idx = 0
dynamic_preds = model.predict_dynamic(
    test_data['x'][sample_idx:sample_idx+1],  # Baseline for one patient
    test_data['z'][sample_idx:sample_idx+1],  # Longitudinal for one patient
    times=[1.0, 2.0, 3.0]  # Times to predict at
)

print(f"\nDynamic predictions for patient {sample_idx}:")
for t, pred in dynamic_preds.items():
    print(f"At t={t:.1f} years: Survival probability = {pred[0]:.2f}")