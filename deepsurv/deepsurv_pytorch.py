import numpy as np
import time
import json
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from lifelines.utils import concordance_index

class DeepSurvDataset(Dataset):
    """Dataset class for DeepSurv model"""
    def __init__(self, x, t, e):
        self.x = torch.FloatTensor(x)
        self.t = torch.FloatTensor(t)
        self.e = torch.IntTensor(e)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.t[idx], self.e[idx]

class DeepSurvLogger:
    """Logger class for DeepSurv model training"""
    def __init__(self, name):
        self.name = name
        self.history = {'loss': [], 'c-index': [], 'valid_loss': [], 'valid_c-index': [], 'lr': []}
    
    def logValue(self, key, value, epoch):
        if key not in self.history:
            self.history[key] = []
        
        while len(self.history[key]) <= epoch:
            self.history[key].append(None)
        
        self.history[key][epoch] = value
    
    def print_progress_bar(self, epoch, n_epochs, loss, ci):
        print(f"Epoch {epoch}/{n_epochs} - Loss: {loss:.4f} - CI: {ci:.4f}")
    
    def logMessage(self, message):
        print(message)
    
    def shutdown(self):
        pass

class NegativeLogLikelihood(nn.Module):
    """
    Negative Log Likelihood loss for survival analysis
    """
    def __init__(self):
        super(NegativeLogLikelihood, self).__init__()
    
    def forward(self, risk, e):
        """
        Parameters:
            risk: (n) risk scores
            e: (n) event indicators
        
        Returns:
            loss: negative partial log-likelihood
        """
        hazard_ratio = torch.exp(risk)
        
        # Sort by risk score (descending) for proper cumulative sum calculation
        sorted_indices = torch.argsort(risk, descending=True)
        hazard_ratio = hazard_ratio[sorted_indices]
        e = e[sorted_indices]
        
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk[sorted_indices] - log_risk
        censored_likelihood = uncensored_likelihood * e
        num_observed_events = torch.sum(e)
        
        # Return negative average log-likelihood
        return -torch.sum(censored_likelihood) / num_observed_events

class DeepSurvNetwork(nn.Module):
    """
    Neural network model for survival analysis
    """
    def __init__(self, n_in, hidden_layers_sizes, activation="relu", dropout=None, batch_norm=False):
        super(DeepSurvNetwork, self).__init__()
        
        # Create layers
        layers = []
        prev_size = n_in
        
        # Set activation function
        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "selu":
            activation_fn = nn.SELU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
        # Create hidden layers
        for n_layer in (hidden_layers_sizes or []):
            layers.append(nn.Linear(prev_size, n_layer))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(n_layer))
                
            layers.append(activation_fn)
            
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_size = n_layer
        
        # Output layer - linear activation
        layers.append(nn.Linear(prev_size, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through the network
        
        Parameters:
            x: (batch_size, n_features) input data
            
        Returns:
            risk: (batch_size, 1) risk scores
        """
        return self.model(x).squeeze()

class DeepSurv:
    def __init__(self, n_in,
                 learning_rate, hidden_layers_sizes=None,
                 lr_decay=0.0, momentum=0.9,
                 L2_reg=0.0, L1_reg=0.0,
                 activation="relu",
                 dropout=None,
                 batch_norm=False,
                 standardize=False,
                 device=None):
        """
        This class implements and trains a DeepSurv model.

        Parameters:
            n_in: number of input nodes.
            learning_rate: learning rate for training.
            lr_decay: coefficient for Power learning rate decay.
            L2_reg: coefficient for L2 weight decay regularization.
            L1_reg: coefficient for L1 weight decay regularization
            momentum: coefficient for momentum. Can be 0 or None to disable.
            hidden_layer_sizes: a list of integers to determine the size of
                each hidden layer.
            activation: activation function.
                Default: "relu"
            batch_norm: True or False. Include batch normalization layers.
            dropout: if not None or 0, the percentage of dropout to include
                after each hidden layer. Default: None
            standardize: True or False. Whether to standardize input features.
            device: torch device to use. If None, will use CUDA if available.
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Default Standardization Values: mean = 0, std = 1
        self.offset = np.zeros(shape=n_in, dtype=np.float32)
        self.scale = np.ones(shape=n_in, dtype=np.float32)
        
        # Create network
        self.network = DeepSurvNetwork(
            n_in=n_in,
            hidden_layers_sizes=hidden_layers_sizes,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        ).to(self.device)
        
        # Create loss function
        self.loss_fn = NegativeLogLikelihood()
        
        # Create optimizer - will be set during training
        self.optimizer = None
        self.scheduler = None
        
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
        
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.standardize = standardize
        
    def _standardize_x(self, x):
        """Standardize input data"""
        return (x - self.offset) / self.scale
        
    def prepare_data(self, dataset):
        """Prepare data for training or evaluation"""
        if isinstance(dataset, dict):
            x, e, t = dataset['x'], dataset['e'], dataset['t']
        
        if self.standardize:
            x = self._standardize_x(x)
        
        # Sort Training Data for Accurate Likelihood (descending order by time)
        sort_idx = np.argsort(t)[::-1]
        x = x[sort_idx]
        e = e[sort_idx]
        t = t[sort_idx]
        
        return x, e, t
    
    def risk(self, x):
        """
        Calculate risk scores for input data
        
        Parameters:
            x: input data
            
        Returns:
            risk scores
        """
        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            risk = self.network(x_tensor)
        return risk
    
    def predict_risk(self, x):
        """
        Predict risk scores for input data (numpy array)
        
        Parameters:
            x: (n,d) numpy array of observations
            
        Returns:
            risks: (n) array of predicted risks
        """
        if self.standardize:
            x = self._standardize_x(x)
            
        self.network.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            risk = self.network(x_tensor)
        return risk.cpu().numpy()
    
    def get_concordance_index(self, x, t, e):
        """
        Calculate concordance index
        
        Parameters:
            x: (n, d) numpy array of observations
            t: (n) numpy array of event times
            e: (n) numpy array of event indicators
            
        Returns:
            concordance index
        """
        if self.standardize:
            x = self._standardize_x(x)
            
        # Get predicted risk scores
        risk_scores = self.predict_risk(x)
        
        # Calculate concordance index (negative because higher risk = lower survival time)
        return concordance_index(t, -risk_scores, e)
    
    def train(self,
              train_data, valid_data=None,
              n_epochs=500,
              batch_size=None,
              validation_frequency=250,
              patience=2000, improvement_threshold=0.99999, patience_increase=2,
              logger=None,
              optimizer=None,
              verbose=True):
        """
        Trains a DeepSurv network on the provided training data and evaluates
        it on the validation data.

        Parameters:
            train_data: dictionary with keys:
                'x': (n,d) array of observations
                't': (n) array of observed time events
                'e': (n) array of observed time indicators
            valid_data: optional. A dictionary with the same keys as train_data.
            n_epochs: maximum number of epochs to train for.
            batch_size: batch size for training. If None, uses all data.
            validation_frequency: how often to compute validation metrics.
            patience: minimum number of epochs to train for.
            improvement_threshold: percentage of improvement needed to increase patience.
            patience_increase: multiplier to patience if threshold is reached.
            logger: None or DeepSurvLogger.
            optimizer: PyTorch optimizer. If None, uses SGD with momentum.
            verbose: whether to print progress.

        Returns:
            metrics: dictionary of training metrics
        """
        if logger is None:
            logger = DeepSurvLogger('DeepSurv')
            
        # Set standardization layer parameters to training data mean and std
        if self.standardize:
            self.offset = train_data['x'].mean(axis=0)
            self.scale = train_data['x'].std(axis=0)
            
        # Prepare training data
        x_train, e_train, t_train = self.prepare_data(train_data)
        train_dataset = DeepSurvDataset(x_train, t_train, e_train)
        
        # Create data loader
        if batch_size is None:
            batch_size = len(x_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Prepare validation data if provided
        if valid_data:
            x_valid, e_valid, t_valid = self.prepare_data(valid_data)
            
        # Create optimizer
        if optimizer is None:
            self.optimizer = optim.SGD(
                self.network.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.L2_reg
            )
        else:
            self.optimizer = optimizer
            
        # Initialize metrics
        best_validation_loss = float('inf')
        best_params = None
        best_params_idx = -1
        
        # Training loop
        start = time.time()
        for epoch in range(n_epochs):
            # Update learning rate with power decay
            lr = self.learning_rate / (1 + epoch * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            logger.logValue('lr', lr, epoch)
            
            # Train for one epoch
            self.network.train()
            total_loss = 0
            for batch_x, batch_t, batch_e in train_loader:
                batch_x = batch_x.to(self.device)
                batch_e = batch_e.to(self.device)
                
                # Forward pass
                risk = self.network(batch_x)
                
                # Calculate loss
                loss = self.loss_fn(risk, batch_e)
                
                # Add L1 regularization
                if self.L1_reg > 0:
                    l1_loss = 0
                    for param in self.network.parameters():
                        l1_loss += torch.sum(torch.abs(param))
                    loss += self.L1_reg * l1_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Average loss for the epoch
            avg_loss = total_loss / len(train_loader)
            logger.logValue('loss', avg_loss, epoch)
            
            # Calculate training concordance index
            ci_train = self.get_concordance_index(x_train, t_train, e_train)
            logger.logValue('c-index', ci_train, epoch)
            
            # Validation
            if valid_data and (epoch % validation_frequency == 0):
                self.network.eval()
                with torch.no_grad():
                    x_valid_tensor = torch.FloatTensor(x_valid).to(self.device)
                    e_valid_tensor = torch.IntTensor(e_valid).to(self.device)
                    
                    # Calculate validation loss
                    valid_risk = self.network(x_valid_tensor)
                    validation_loss = self.loss_fn(valid_risk, e_valid_tensor).item()
                    logger.logValue('valid_loss', validation_loss, epoch)
                    
                    # Calculate validation concordance index
                    ci_valid = self.get_concordance_index(x_valid, t_valid, e_valid)
                    logger.logValue('valid_c-index', ci_valid, epoch)
                    
                    # Check if this is the best model so far
                    if validation_loss < best_validation_loss:
                        # Improve patience if loss improves enough
                        if validation_loss < best_validation_loss * improvement_threshold:
                            patience = max(patience, epoch * patience_increase)
                        
                        best_params = self.network.state_dict().copy()
                        best_params_idx = epoch
                        best_validation_loss = validation_loss
            
            # Print progress
            if verbose and (epoch % validation_frequency == 0):
                logger.print_progress_bar(epoch, n_epochs, avg_loss, ci_train)
            
            # Early stopping
            if patience <= epoch:
                break
        
        if verbose:
            logger.logMessage(f'Finished Training with {epoch + 1} iterations in {time.time() - start:.2f}s')
        logger.shutdown()
        
        # Update logger history with best model info
        logger.history['best_valid_loss'] = best_validation_loss
        logger.history['best_params'] = best_params
        logger.history['best_params_idx'] = best_params_idx
        
        return logger.history
    
    def to_json(self):
        """Convert model hyperparameters to JSON string"""
        return json.dumps(self.hyperparams)
    
    def save_model(self, filename, weights_file=None):
        """Save model hyperparameters to JSON file"""
        with open(filename, 'w') as fp:
            fp.write(self.to_json())
        
        if weights_file:
            self.save_weights(weights_file)
    
    def save_weights(self, filename):
        """Save model weights to file"""
        torch.save({
            'state_dict': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'offset': self.offset,
            'scale': self.scale
        }, filename)
    
    def load_weights(self, filename):
        """Load model weights from file"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.network.load_state_dict(checkpoint['state_dict'])
        if checkpoint['optimizer'] and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.offset = checkpoint['offset']
        self.scale = checkpoint['scale']
    
    def recommend_treatment(self, x, trt_i, trt_j, trt_idx=-1):
        """
        Computes recommendation function rec_ij(x) for two treatments i and j.
        rec_ij(x) is the log of the hazards ratio of x in treatment i vs.
        treatment j.

        Parameters:
            x: (n, d) numpy array of observations
            trt_i: treatment i value
            trt_j: treatment j value
            trt_idx: the index of x representing the treatment group column

        Returns:
            rec_ij: recommendation (difference in risk)
        """
        # Copy x to prevent overwriting data
        x_trt = np.copy(x)
        
        # Calculate risk of observations with treatment i
        x_trt[:, trt_idx] = trt_i
        h_i = self.predict_risk(x_trt)
        
        # Risk of observations with treatment j
        x_trt[:, trt_idx] = trt_j
        h_j = self.predict_risk(x_trt)
        
        rec_ij = h_i - h_j
        return rec_ij
    
    def plot_risk_surface(self, data, i=0, j=1,
                          figsize=(6, 4), x_lims=None, y_lims=None, c_lims=None):
        """
        Plots the predicted risk surface of the network with respect to two
        observed covariates i and j.

        Parameters:
            data: (n,d) numpy array of observations of which to predict risk.
            i: index of data to plot as axis 1
            j: index of data to plot as axis 2
            figsize: size of figure for matplotlib
            x_lims: Optional. If provided, override default x_lims
            y_lims: Optional. If provided, override default y_lims
            c_lims: Optional. If provided, override default color limits.

        Returns:
            fig: matplotlib figure object.
        """
        fig = plt.figure(figsize=figsize)
        X = data[:, i]
        Y = data[:, j]
        Z = self.predict_risk(data)
        
        if x_lims is None:
            x_lims = [np.round(np.min(X)), np.round(np.max(X))]
        if y_lims is None:
            y_lims = [np.round(np.min(Y)), np.round(np.max(Y))]
        if c_lims is None:
            c_lims = [np.round(np.min(Z)), np.round(np.max(Z))]
        
        ax = plt.scatter(X, Y, c=Z, edgecolors='none', marker='.')
        ax.set_clim(*c_lims)
        plt.colorbar()
        plt.xlim(*x_lims)
        plt.ylim(*y_lims)
        plt.xlabel('$x_{%d}$' % i, fontsize=18)
        plt.ylabel('$x_{%d}$' % j, fontsize=18)
        
        return fig

def load_model_from_json(model_fp, weights_fp=None, device=None):
    """
    Load a DeepSurv model from a JSON file and optionally load weights.
    
    Parameters:
        model_fp: path to JSON model file
        weights_fp: optional path to weights file
        device: torch device to use (defaults to CUDA if available)
        
    Returns:
        model: loaded DeepSurv model
    """
    with open(model_fp, 'r') as fp:
        json_model = fp.read()
    print('Loading json model:', json_model)
    hyperparams = json.loads(json_model)
    
    # Add device parameter
    if device is not None:
        hyperparams['device'] = device
    
    model = DeepSurv(**hyperparams)
    
    if weights_fp:
        model.load_weights(weights_fp)
    
    return model