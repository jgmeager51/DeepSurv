import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
import h5py
from typing import Optional, Dict, List, Tuple
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from deep_datasets import SimulatedData  # If using as a module
from sklearn.model_selection import train_test_split  # This was missing


class DeepSurv(nn.Module):
    def __init__(self, n_in: int,
                 learning_rate: float = 0.01,
                 hidden_layers_sizes: Optional[List[int]] = None,
                 lr_decay: float = 0.0,
                 momentum: float = 0.9,
                 L2_reg: float = 0.0,
                 L1_reg: float = 0.0,
                 activation: str = "rectify",
                 dropout: Optional[float] = None,
                 batch_norm: bool = False,
                 standardize: bool = False):
        """
        PyTorch implementation of DeepSurv model.
        
        Parameters:
            n_in: Number of input features
            learning_rate: Initial learning rate
            hidden_layers_sizes: List of hidden layer sizes
            lr_decay: Learning rate decay coefficient
            momentum: Momentum coefficient
            L2_reg: L2 regularization coefficient
            L1_reg: L1 regularization coefficient
            activation: Activation function ('rectify' or 'selu')
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
            standardize: Whether to standardize input data
        """
        super().__init__()
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.standardize = standardize
        
        # Standardization parameters
        self.register_buffer('offset', torch.zeros(n_in))
        self.register_buffer('scale', torch.ones(n_in))
        
        # Build network
        layers = []
        input_dim = n_in
        
        if hidden_layers_sizes is None:
            hidden_layers_sizes = []
            
        for layer_size in hidden_layers_sizes:
            layers.append(nn.Linear(input_dim, layer_size))
            
            # Activation
            if activation == 'rectify':
                layers.append(nn.ReLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
                
            # Batch norm
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_size))
                
            # Dropout
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            input_dim = layer_size
            
        # Final output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Store hyperparameters
        self.hyperparams = {
            'n_in': n_in,
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        if self.standardize:
            x = (x - self.offset) / self.scale
        return self.network(x).squeeze()
    
    def _negative_log_likelihood(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Compute Cox partial likelihood loss"""
        risk = self(x)
        hazard_ratio = torch.exp(risk)
        
        # Sort by time descending
        _, idx = torch.sort(-x)
        risk = risk[idx]
        hazard_ratio = hazard_ratio[idx]
        e = e[idx]
        
        # Compute log sum risk
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        
        # Compute likelihood
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * e
        num_observed_events = torch.sum(e)
        
        # Negative log likelihood with regularization
        neg_likelihood = -torch.sum(censored_likelihood) / num_observed_events
        return neg_likelihood
    
    def get_concordance_index(self, x: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
        """Compute concordance index"""
        with torch.no_grad():
            risk = self(torch.FloatTensor(x)).numpy()
        return concordance_index(t, -risk, e)
    
    def prepare_data(self, dataset: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training"""
        x, e, t = dataset['x'], dataset['e'], dataset['t']
        
        if self.standardize:
            x = (x - self.offset.numpy()) / self.scale.numpy()
            
        # Sort by time descending
        sort_idx = np.argsort(t)[::-1]
        x = x[sort_idx]
        e = e[sort_idx]
        t = t[sort_idx]
        
        return (
            torch.FloatTensor(x),
            torch.FloatTensor(e),
            torch.FloatTensor(t)
        )
    
    def train_model(self,
                   train_data: Dict[str, np.ndarray],
                   valid_data: Optional[Dict[str, np.ndarray]] = None,
                   n_epochs: int = 500,
                   validation_frequency: int = 250,
                   patience: int = 2000,
                   improvement_threshold: float = 0.99999,
                   patience_increase: int = 2,
                   verbose: bool = True) -> Dict:
        """
        Train the DeepSurv model.
        
        Returns:
            Dictionary containing training history and metrics
        """
        # Set standardization parameters
        if self.standardize:
            self.offset = torch.FloatTensor(train_data['x'].mean(axis=0))
            self.scale = torch.FloatTensor(train_data['x'].std(axis=0))
        
        # Prepare data
        x_train, e_train, t_train = self.prepare_data(train_data)
        if valid_data is not None:
            x_valid, e_valid, t_valid = self.prepare_data(valid_data)
        
        # Initialize optimizer
        optimizer = optim.SGD(self.parameters(),
                            lr=self.learning_rate,
                            momentum=self.momentum,
                            weight_decay=self.L2_reg)
        
        # Training metrics
        best_validation_loss = np.inf
        best_params = None
        best_params_idx = -1
        history = {
            'train_loss': [],
            'train_ci': [],
            'valid_loss': [],
            'valid_ci': [],
            'best_params': None,
            'best_params_idx': -1,
            'best_valid_loss': np.inf,
            'best_valid_ci': 0.0
        }
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            # Learning rate decay
            lr = self.learning_rate / (1 + epoch * self.lr_decay)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            # Training step
            self.train()
            optimizer.zero_grad()
            loss = self._negative_log_likelihood(x_train, e_train)
            
            # Add L1 regularization
            if self.L1_reg > 0:
                l1_reg = torch.tensor(0.)
                for param in self.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += self.L1_reg * l1_reg
                
            loss.backward()
            optimizer.step()
            
            # Store metrics
            history['train_loss'].append(loss.item())
            ci_train = self.get_concordance_index(x_train.numpy(), t_train.numpy(), e_train.numpy())
            history['train_ci'].append(ci_train)
            
            # Validation
            if valid_data is not None and (epoch % validation_frequency == 0):
                self.eval()
                with torch.no_grad():
                    valid_loss = self._negative_log_likelihood(x_valid, e_valid).item()
                    ci_valid = self.get_concordance_index(x_valid.numpy(), t_valid.numpy(), e_valid.numpy())
                
                history['valid_loss'].append(valid_loss)
                history['valid_ci'].append(ci_valid)
                
                if valid_loss < best_validation_loss:
                    # Improve patience if loss improves enough
                    if valid_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, epoch * patience_increase)
                    
                    best_params = [p.detach().clone() for p in self.parameters()]
                    best_params_idx = epoch
                    best_validation_loss = valid_loss
                    history['best_valid_ci'] = ci_valid
                    
            if verbose and (epoch % validation_frequency == 0):
                print(f"Epoch {epoch}/{n_epochs} - Loss: {loss.item():.4f} - Train CI: {ci_train:.4f}", end="")
                if valid_data is not None:
                    print(f" - Val Loss: {valid_loss:.4f} - Val CI: {ci_valid:.4f}")
                else:
                    print()
            
            if patience <= epoch:
                break
                
        if verbose:
            print(f"Training completed in {time.time() - start_time:.2f}s")
            
        # Store best parameters
        history['best_params'] = best_params
        history['best_params_idx'] = best_params_idx
        history['best_valid_loss'] = best_validation_loss
        
        return history
    
    def predict_risk(self, x: np.ndarray) -> np.ndarray:
        """Predict risk scores for input data"""
        self.eval()
        with torch.no_grad():
            return self(torch.FloatTensor(x)).numpy()
        
    def recommend_treatment(self, x: np.ndarray, trt_i: float, trt_j: float, trt_idx: int = -1) -> np.ndarray:
        """Compute treatment recommendation scores"""
        x_trt = x.copy()
        
        # Risk with treatment i
        x_trt[:, trt_idx] = trt_i
        h_i = self.predict_risk(x_trt)
        
        # Risk with treatment j
        x_trt[:, trt_idx] = trt_j
        h_j = self.predict_risk(x_trt)
        
        return h_i - h_j
    
    def save_model(self, filename: str, weights_file: Optional[str] = None):
        """Save model configuration to JSON file"""
        with open(filename, 'w') as fp:
            json.dump(self.hyperparams, fp)
            
        if weights_file:
            torch.save(self.state_dict(), weights_file)
            
    @classmethod
    def load_model(cls, model_file: str, weights_file: Optional[str] = None):
        """Load model from configuration file"""
        with open(model_file, 'r') as fp:
            hyperparams = json.load(fp)
            
        model = cls(**hyperparams)
        if weights_file:
            model.load_state_dict(torch.load(weights_file))
            
        return model
    

# Example usage with DeepSurv training:
# In your prepare_deepsurv_data function, consider adding:
def prepare_deepsurv_data(hr_ratio=2.0, n_samples=1000, test_size=0.2,
                         n_features=10, treatment_group=True, random_state=42):
    """Generates and prepares data with improved validation"""
    try:
        data_gen = SimulatedData(
            hr_ratio=hr_ratio,
            num_features=n_features,
            treatment_group=treatment_group
        )
        full_data = data_gen.generate_data(N=n_samples, method='linear')
        
        # Validate data shapes
        assert full_data['x'].shape == (n_samples, n_features + int(treatment_group))
        assert len(full_data['t']) == n_samples
        assert len(full_data['e']) == n_samples
        
        # Split data
        X_train, X_test, t_train, t_test, e_train, e_test = train_test_split(
            full_data['x'], full_data['t'], full_data['e'], 
            test_size=test_size, random_state=random_state
        )

        
        return {
            'train': {'x': X_train.astype(np.float32),
                     't': t_train.astype(np.float32),
                     'e': e_train.astype(np.int32)},
            'test': {'x': X_test.astype(np.float32),
                    't': t_test.astype(np.float32),
                    'e': e_test.astype(np.int32)}
        }
    except Exception as e:
        print(f"Data generation failed: {str(e)}")
        raise
# Example usage:
if __name__ == "__main__":
    # Generate data
    train_data, test_data = prepare_deepsurv_data()
    
    # Initialize model (using the PyTorch DeepSurv we created)
    model = DeepSurv(
        n_in=train_data['x'].shape[1],
        hidden_layers_sizes=[100, 50],
        activation='rectify',
        dropout=0.1,
        batch_norm=True
    )
    
    # Train model
    history = model.train_model(
        train_data=train_data,
        valid_data=test_data,
        n_epochs=500,
        validation_frequency=50
    )
    
    # Evaluate on test data
    test_ci = model.get_concordance_index(
        test_data['x'], 
        test_data['t'], 
        test_data['e']
    )
    print(f"\nFinal Test Concordance Index: {test_ci:.4f}")