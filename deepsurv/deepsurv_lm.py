import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from copy import deepcopy
from sksurv.metrics import cumulative_dynamic_auc  # Import for time-dependent AUC

from deepsurv_pytorch import DeepSurv, DeepSurvDataset, DeepSurvLogger,DeepSurvNetwork  # Assuming DeepSurv is defined in deep_surv.py

class LandmarkDeepSurv:
    def __init__(self, n_baseline_features, n_longitudinal_features, landmark_times, prediction_window,
                 learning_rate=1e-3, hidden_layers_sizes=None,
                 lr_decay=0.0, momentum=0.9,
                 L2_reg=0.0, L1_reg=0.0,
                 activation="relu",
                 dropout=None,
                 batch_norm=False,
                 standardize=False,
                 device=None):
        """
        Landmark DeepSurv model for dynamic prediction with longitudinal data.
        
        Parameters:
            n_baseline_features: number of baseline covariates
            n_longitudinal_features: number of longitudinal markers
            landmark_times: list of landmark times for dynamic prediction
            prediction_window: prediction window after each landmark time
            learning_rate: learning rate for training
            hidden_layers_sizes: list of hidden layer sizes
            lr_decay: learning rate decay coefficient
            momentum: momentum for SGD
            L2_reg: L2 regularization coefficient
            L1_reg: L1 regularization coefficient
            activation: activation function ('relu' or 'selu')
            dropout: dropout rate
            batch_norm: whether to use batch normalization
            standardize: whether to standardize input features
            device: device to use ('cpu' or 'cuda')
        """
        self.n_baseline_features = n_baseline_features
        self.n_longitudinal_features = n_longitudinal_features
        self.n_in = n_baseline_features + n_longitudinal_features  # Total input features
        self.landmark_times = np.array(landmark_times)
        self.prediction_window = prediction_window
        self.models = {}  # Dictionary to store models for each landmark time
        
        # Hyperparameters for individual DeepSurv models
        self.hyperparams = {
            'n_in': self.n_in,
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize,
            'device': device
        }
    
    def _prepare_landmark_dataset(self, data, landmark_time):
        """
        Prepare dataset for a specific landmark time.
        
        Parameters:
            data: dictionary containing:
                - 'x': baseline covariates (n_samples, n_features)
                - 'z': longitudinal measurements (n_samples, n_timepoints, n_markers)
                - 't': event times (n_samples)
                - 'e': event indicators (n_samples)
            landmark_time: current landmark time
            
        Returns:
            Dictionary containing prepared dataset for the landmark time
        """
        x = data['x']
        z = data['z']
        t = data['t']
        e = data['e']
        
        # Select subjects at risk at landmark_time (T_i > landmark_time)
        at_risk = t > landmark_time
        x_landmark = x[at_risk]
        t_landmark = t[at_risk]
        e_landmark = e[at_risk]
        
        # Get the most recent longitudinal measurement before landmark_time (LOCF)
        z_landmark = []
        for i in range(len(z)):
            if at_risk[i]:
                # Find measurements before or at landmark_time
                valid_measurements = [m for m in z[i] if m['time'] <= landmark_time]
                if valid_measurements:
                    # Get the most recent measurement (LOCF)
                    latest_measurement = max(valid_measurements, key=lambda m: m['time'])
                    z_landmark.append(latest_measurement['values'])
                else:
                    # If no measurements, use zeros (could also use mean imputation)
                    z_landmark.append(np.zeros(z[0][0]['values'].shape))
        
        z_landmark = np.array(z_landmark)
        
        # Combine baseline and longitudinal features
        xz_landmark = np.concatenate([x_landmark, z_landmark], axis=1)
        
        # Administrative censoring at landmark_time + prediction_window
        t_landmark_censored = np.minimum(t_landmark, landmark_time + self.prediction_window)
        e_landmark_censored = e_landmark.copy()
        e_landmark_censored[t_landmark > landmark_time + self.prediction_window] = 0
        
        return {
            'x': xz_landmark,
            't': t_landmark_censored,
            'e': e_landmark_censored,
            'original_t': t_landmark,
            'original_e': e_landmark
        }
    def _compute_feature_importance(self, landmark_time, landmark_data):
        """
        Compute feature importance for a specific landmark time.
        
        Parameters:
            landmark_time: The landmark time for which to compute feature importance.
            landmark_data: The dataset for the landmark time.
            """
        model = self.models[landmark_time]
        baseline_risk = model.predict_risk(landmark_data['x'])
        feature_importance = []

        for i in range(landmark_data['x'].shape[1]):
            x_perturbed = landmark_data['x'].copy()
            x_perturbed[:, i] += np.std(x_perturbed[:, i])  # Perturb feature by 1 std
            perturbed_risk = model.predict_risk(x_perturbed)
            importance = np.mean(np.abs(perturbed_risk - baseline_risk))
            feature_importance.append(importance)

        self.feature_importances[landmark_time] = feature_importance
    def train(self, data, n_epochs=500, batch_size=None, validation_frequency=50, verbose=True):
        """
        Train Landmark DeepSurv models for each landmark time.
        """
        self.feature_importances = {}
        
        for landmark_time in self.landmark_times:
            if verbose:
                print(f"\nTraining model for landmark time {landmark_time}")
            
            # Prepare dataset for this landmark time
            landmark_data = self._prepare_landmark_dataset(data, landmark_time)
            
            # Split into training and validation sets (80/20)
            n_samples = len(landmark_data['x'])
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            split = int(0.8 * n_samples)
            
            train_idx, valid_idx = indices[:split], indices[split:]
            
            train_data = {
                'x': landmark_data['x'][train_idx],
                't': landmark_data['t'][train_idx],
                'e': landmark_data['e'][train_idx]
            }
            
            valid_data = {
                'x': landmark_data['x'][valid_idx],
                't': landmark_data['t'][valid_idx],
                'e': landmark_data['e'][valid_idx]
            }
            
            # Create and train DeepSurv model for this landmark time
            model = DeepSurv(**self.hyperparams)
            
            # Convert to PyTorch datasets
            train_dataset = DeepSurvDataset(train_data['x'], train_data['t'], train_data['e'])
            valid_dataset = DeepSurvDataset(valid_data['x'], valid_data['t'], valid_data['e'])
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size or len(train_data['x']), shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size or len(valid_data['x']), shuffle=False)
            
            # Training loop
            optimizer = optim.Adam(model.network.parameters(), lr=self.hyperparams['learning_rate'])
            
            for epoch in range(n_epochs):
                model.network.train()
                total_loss = 0
                
                for batch_x, batch_t, batch_e in train_loader:
                    batch_x = batch_x.to(model.device)
                    batch_t = batch_t.to(model.device)
                    batch_e = batch_e.to(model.device)
                    
                    # Forward pass
                    risk = model.network(batch_x)
                    
                    # Calculate loss - now passing all three required arguments
                    loss = model.loss_fn(risk, batch_e, batch_t)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                # Validation
                if valid_data and (epoch % validation_frequency == 0 or epoch == n_epochs - 1):
                    model.network.eval()
                    valid_loss = 0
                    with torch.no_grad():
                        for batch_x, batch_t, batch_e in valid_loader:
                            batch_x = batch_x.to(model.device)
                            batch_t = batch_t.to(model.device)
                            batch_e = batch_e.to(model.device)
                            
                            valid_risk = model.network(batch_x)
                            valid_loss += model.loss_fn(valid_risk, batch_e, batch_t).item()
                    
                    if verbose:
                        print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {total_loss/len(train_loader):.4f} - Valid Loss: {valid_loss/len(valid_loader):.4f}")
            
            # Store the trained model
            self.models[landmark_time] = model
            
            # Compute baseline hazard for survival predictions
            model.compute_baseline_hazard(
                train_data['x'],
                train_data['t'],
                train_data['e']
            )
            
            # Compute feature importance for this landmark time
            self._compute_feature_importance(landmark_time, landmark_data)
        
    def predict_survival(self, x, z, landmark_time, prediction_times=None):
        """
        Predict survival probability at a given landmark time.
        
        Parameters:
            x: baseline covariates (n_samples, n_features)
            z: longitudinal measurements (n_samples, n_timepoints, n_markers)
            landmark_time: landmark time for prediction
            prediction_times: specific times to predict at (defaults to up to prediction_window)
            
        Returns:
            survival probabilities (n_samples, n_prediction_times)
        """
        if landmark_time not in self.models:
            raise ValueError(f"No model trained for landmark time {landmark_time}")
        
        model = self.models[landmark_time]
        
        # Prepare input data for this landmark time (LOCF imputation)
        xz_landmark = []
        for i in range(len(x)):
            # Find most recent measurement before landmark_time
            valid_measurements = [m for m in z[i] if m['time'] <= landmark_time]
            if valid_measurements:
                latest_measurement = max(valid_measurements, key=lambda m: m['time'])
                z_i = latest_measurement['values']
            else:
                z_i = np.zeros(z[0][0]['values'].shape)  # Default to zeros if no measurements
            
            xz_landmark.append(np.concatenate([x[i], z_i]))
        
        xz_landmark = np.array(xz_landmark)
        
        # If no specific prediction times given, use a grid up to prediction_window
        if prediction_times is None:
            prediction_times = np.linspace(
                landmark_time,
                landmark_time + self.prediction_window,
                100
            )
        
        # Predict survival probabilities
        survival_probs = model.predict_survival(xz_landmark, prediction_times)
        
        return survival_probs
    
    def predict_dynamic(self, x, z, times):
        """
        Make dynamic predictions over time.
        
        Parameters:
            x: baseline covariates (n_samples, n_features)
            z: longitudinal measurements (n_samples, n_timepoints, n_markers)
            times: time points at which to make predictions
            
        Returns:
            Dictionary of survival probabilities for each time point
        """
        predictions = {}
        
        for t in times:
            # Find the most recent landmark time before t
            landmark_times_before = self.landmark_times[self.landmark_times <= t]
            if len(landmark_times_before) == 0:
                continue  # No model trained for this early time
            
            landmark_time = max(landmark_times_before)
            
            # Predict survival from this landmark time
            prediction_window = min(self.prediction_window, t - landmark_time)
            pred_times = np.linspace(landmark_time, landmark_time + prediction_window, 10)
            
            survival_probs = self.predict_survival(x, z, landmark_time, pred_times)
            
            # Store the prediction at time t (last time point)
            predictions[t] = survival_probs[:, -1]
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get feature importance scores for all landmark times.
        
        Returns:
            Dictionary of feature importance scores for each landmark time
        """
        return self.feature_importances
    
    def evaluate(self, data, metrics=['c-index', 'brier', 'auc']):
        """
        Evaluate model performance at each landmark time.
        
        Parameters:
            data: dictionary containing:
                - 'x': baseline covariates (n_samples, n_features)
                - 'z': longitudinal measurements (n_samples, n_timepoints, n_markers)
                - 't': event times (n_samples)
                - 'e': event indicators (n_samples)
            metrics: list of metrics to compute ('c-index', 'brier')
            
        Returns:
            Dictionary of evaluation results for each landmark time
        """
        results = {}
        
        for landmark_time in self.landmark_times:
            # Prepare dataset for this landmark time
            landmark_data = self._prepare_landmark_dataset(data, landmark_time)
            model = self.models[landmark_time]
            
            x = landmark_data['x']
            t = landmark_data['t']
            e = landmark_data['e']
            original_t = landmark_data['original_t']
            original_e = landmark_data['original_e']
            
            landmark_results = {}
            
            if 'c-index' in metrics:
                ci = model.get_concordance_index(x, t, e)
                landmark_results['c-index'] = ci
            
            if 'brier' in metrics:
                # Compute Brier score at landmark_time + prediction_window/2
                pred_time = landmark_time + self.prediction_window / 2
                survival_probs = model.predict_survival(x, [pred_time])
                
                # Event indicator at pred_time
                event_indicator = (original_t <= pred_time) & (original_e == 1)
                
                # Brier score
                brier_score = np.mean((event_indicator - (1 - survival_probs.squeeze()))**2)
                landmark_results['brier'] = brier_score
                # Time-dependent AUC
            if 'auc' in metrics:
                 # Restrict time grid to the range of observed times in the test data
                min_time = np.min(original_t)
                max_time = np.max(original_t)
                time_grid = np.linspace(max(min_time, landmark_time), min(max_time, landmark_time + self.prediction_window), 100)

                # Convert event indicators and times to structured array for sksurv
                structured_events = np.array([(bool(ei), ti) for ei, ti in zip(original_e, original_t)],
                                            dtype=[('event', bool), ('time', float)])

                # Predict survival probabilities for the time grid
                survival_probs = model.predict_survival(x, times=time_grid)

                # Compute time-dependent AUC
                auc_times, auc_values = cumulative_dynamic_auc(
                    structured_events, structured_events, 1 - survival_probs, time_grid
                )
                landmark_results['auc'] = {
                    'times': auc_times.tolist(),
                    'values': auc_values.tolist()
                }
                
            results[landmark_time] = landmark_results
        
        return results
    


# The original DeepSurv implementation remains unchanged below
# (all the classes: DeepSurvDataset, DeepSurvLogger, NegativeLogLikelihood, 
# DeepSurvNetwork, and DeepSurv)




# Get feature importance
# feature_names = [
#     'age', 'sex', 'bsa', 'emergenc', 'hs', 'dm', 'hc', 'prenyha', 'lvh', 
#     'creat', 'acei', 'con_cabg', 'sten_reg_mix', 'lv', 'size', 'lvmi', 'grad'
# ]

#print("\nFeature Importances:")
#importances = model._compute_feature_importance()
# for lm_time, importance in importances.items():
#     print(f"\nLandmark {lm_time:.1f} years:")
#     sorted_idx = np.argsort(-importance)  # Sort descending
#     for i in sorted_idx:
#         print(f"{feature_names[i]:<15}: {importance[i]:.4f}")