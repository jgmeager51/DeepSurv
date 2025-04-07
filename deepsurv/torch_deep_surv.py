import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
from torch.utils.data import DataLoader

import sys
sys.path.append("c:/Research/deepsurvtorch/deepsurv/")
from torch_datasets import SimulatedData


class DeepSurv(nn.Module):
    def __init__(self, n_in, hidden_layers_sizes=None, activation="relu", dropout=None, batch_norm=False, standardize=False):
        """
        Initializes the DeepSurv model using PyTorch.

        Parameters:
            n_in: Number of input features.
            hidden_layers_sizes: List of integers specifying the size of each hidden layer.
            activation: Activation function ('relu' or 'selu').
            dropout: Dropout rate (e.g., 0.5 for 50% dropout). Default is None (no dropout).
            batch_norm: Whether to include batch normalization layers. Default is False.
            standardize: Whether to standardize input data. Default is False.
        """
        super(DeepSurv, self).__init__()

        # Define activation function
        if activation == "relu":
            activation_fn = nn.ReLU
        elif activation == "selu":
            activation_fn = nn.SELU
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Build the network layers
        layers = []
        input_size = n_in
        for hidden_size in (hidden_layers_sizes or []):
            layers.append(nn.Linear(input_size, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_size = hidden_size

        # Output layer (linear activation for log hazard ratio)
        layers.append(nn.Linear(input_size, 1))

        # Combine layers into a sequential model
        self.network = nn.Sequential(*layers)

        # Standardization parameters
        self.standardize = standardize
        if standardize:
            self.register_buffer('offset', torch.zeros(n_in))
            self.register_buffer('scale', torch.ones(n_in))

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
            x: Input tensor of shape (batch_size, n_in).

        Returns:
            Output tensor of shape (batch_size, 1) representing log hazard ratios.
        """
        if self.standardize:
            x = (x - self.offset) / self.scale
        return self.network(x).squeeze()

    def negative_log_likelihood(self, x, e):
        """
        Computes the negative partial log-likelihood for Cox proportional hazards.

        Parameters:
            x: Input tensor of shape (batch_size, n_in).
            e: Event indicators tensor of shape (batch_size,) (1 = event, 0 = censored).

        Returns:
            Negative log-likelihood loss.
        """
        log_hazard = self.forward(x).squeeze()
        hazard_ratio = torch.exp(log_hazard)
        log_cumsum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = log_hazard - log_cumsum_hazard
        censored_likelihood = uncensored_likelihood * e
        neg_likelihood = -torch.sum(censored_likelihood) / torch.sum(e)
        return neg_likelihood

    def get_concordance_index(self, x, t, e):
        """
        Calculates the concordance index (C-index) for the model's predictions.

        Parameters:
            x: Input tensor of shape (n, n_in).
            t: Observed time events (numpy array).
            e: Event indicators (numpy array).

        Returns:
            Concordance index (float).
        """
        with torch.no_grad():
            log_hazard = self.forward(x).squeeze().numpy()
        return concordance_index(t, -log_hazard, e)

    def save_model(self, filename, weights_file=None):
        """Save model configuration and weights."""
        with open(filename, 'w') as f:
            json.dump(self.hyperparams, f)
        if weights_file:
            torch.save(self.state_dict(), weights_file)

    @classmethod
    def load_model(cls, model_file, weights_file=None):
        """Load model configuration and weights."""
        with open(model_file, 'r') as f:
            hyperparams = json.load(f)
        model = cls(**hyperparams)
        if weights_file:
            model.load_state_dict(torch.load(weights_file))
        return model


def train_deepsurv(model, train_data, valid_data=None, n_epochs=500, learning_rate=1e-4, weight_decay=1e-4, verbose=True):
    """
    Trains the DeepSurv model.

    Parameters:
        model: DeepSurv model instance.
        train_data: Dictionary with keys 'x', 't', 'e' for training data.
        valid_data: Optional. Dictionary with keys 'x', 't', 'e' for validation data.
        n_epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        verbose: Whether to print training progress.

    Returns:
        Dictionary containing training and validation metrics.
    """
    # Prepare data
    x_train = torch.tensor(train_data['x'], dtype=torch.float32)
    t_train = train_data['t']
    e_train = torch.tensor(train_data['e'], dtype=torch.float32)

    if valid_data:
        x_valid = torch.tensor(valid_data['x'], dtype=torch.float32)
        t_valid = valid_data['t']
        e_valid = torch.tensor(valid_data['e'], dtype=torch.float32)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    metrics = {'train_loss': [], 'train_ci': [], 'valid_loss': [], 'valid_ci': []}
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.negative_log_likelihood(x_train, e_train)
        loss.backward()
        optimizer.step()

        # Compute training metrics
        metrics['train_loss'].append(loss.item())
        train_ci = model.get_concordance_index(x_train, t_train, e_train.numpy())
        metrics['train_ci'].append(train_ci)

        # Validation metrics
        if valid_data:
            model.eval()
            with torch.no_grad():
                valid_loss = model.negative_log_likelihood(x_valid, e_valid)
                metrics['valid_loss'].append(valid_loss.item())
                valid_ci = model.get_concordance_index(x_valid, t_valid, e_valid.numpy())
                metrics['valid_ci'].append(valid_ci)

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Train CI: {train_ci:.4f}")

    return metrics


if __name__ == "__main__":
    # Generate simulated data
    sim_data = SimulatedData(
        N=1000,
        hr_ratio=2.0,
        average_death=5,
        censor_mode='end_time',
        end_time=15,
        num_features=10,
        num_var=2,
        treatment_group=True,
        method='gaussian',
        gaussian_config={'c': 0.0, 'rad': 0.5},
        as_tensor=False
    )

    # Extract data
    x = sim_data.dataset['x']
    t = sim_data.dataset['t']
    e = sim_data.dataset['e']

    # Split data into training and validation sets
    x_train, x_valid, t_train, t_valid, e_train, e_valid = train_test_split(
        x, t, e, test_size=0.2, random_state=42
    )

    # Prepare training and validation dictionaries
    train_data = {'x': x_train, 't': t_train, 'e': e_train}
    valid_data = {'x': x_valid, 't': t_valid, 'e': e_valid}

    # Initialize the model
    model = DeepSurv(
        n_in=11,
        hidden_layers_sizes=[32, 16],
        activation="relu",
        dropout=0.5,
        batch_norm=True,
        standardize=True
    )

    # Train the model
    metrics = train_deepsurv(model, train_data, valid_data, n_epochs=100, learning_rate=1e-3)

    # Plot training metrics
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['valid_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from lifelines.utils import concordance_index

# import sys
# sys.path.append("c:/Research/deepsurvtorch/deepsurv/")
# from torch_datasets import SimulatedData


# class DeepSurv(nn.Module):
#     def __init__(self, n_in, hidden_layers_sizes=None, activation="relu", dropout=None, batch_norm=False):
#         """
#         Initializes the DeepSurv model using PyTorch.

#         Parameters:
#             n_in: Number of input features.
#             hidden_layers_sizes: List of integers specifying the size of each hidden layer.
#             activation: Activation function ('relu' or 'selu').
#             dropout: Dropout rate (e.g., 0.5 for 50% dropout). Default is None (no dropout).
#             batch_norm: Whether to include batch normalization layers. Default is False.
#         """
#         super(DeepSurv, self).__init__()

#         # Define activation function
#         if activation == "relu":
#             activation_fn = nn.ReLU
#         elif activation == "selu":
#             activation_fn = nn.SELU
#         else:
#             raise ValueError(f"Unknown activation function: {activation}")

#         # Build the network layers
#         layers = []
#         input_size = n_in
#         for hidden_size in (hidden_layers_sizes or []):
#             layers.append(nn.Linear(input_size, hidden_size))
#             if batch_norm:
#                 layers.append(nn.BatchNorm1d(hidden_size))
#             layers.append(activation_fn())
#             if dropout:
#                 layers.append(nn.Dropout(dropout))
#             input_size = hidden_size

#         # Output layer (linear activation for log hazard ratio)
#         layers.append(nn.Linear(input_size, 1))

#         # Combine layers into a sequential model
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         Forward pass through the network.

#         Parameters:
#             x: Input tensor of shape (batch_size, n_in).

#         Returns:
#             Output tensor of shape (batch_size, 1) representing log hazard ratios.
#         """
#         return self.network(x)

#     def negative_log_likelihood(self, x, e):
#         """
#         Computes the negative partial log-likelihood for Cox proportional hazards.

#         Parameters:
#             x: Input tensor of shape (batch_size, n_in).
#             e: Event indicators tensor of shape (batch_size,) (1 = event, 0 = censored).

#         Returns:
#             Negative log-likelihood loss.
#         """
#         log_hazard = self.forward(x).squeeze()
#         hazard_ratio = torch.exp(log_hazard)
#         log_cumsum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))
#         uncensored_likelihood = log_hazard - log_cumsum_hazard
#         censored_likelihood = uncensored_likelihood * e
#         neg_likelihood = -torch.sum(censored_likelihood) / torch.sum(e)
#         return neg_likelihood

#     def get_concordance_index(self, x, t, e):
#         """
#         Calculates the concordance index (C-index) for the model's predictions.

#         Parameters:
#             x: Input tensor of shape (n, n_in).
#             t: Observed time events (numpy array).
#             e: Event indicators (numpy array).

#         Returns:
#             Concordance index (float).
#         """
#         with torch.no_grad():
#             log_hazard = self.forward(x).squeeze().numpy()
#         return concordance_index(t, -log_hazard, e)


# def train_deepsurv(model, train_data, valid_data=None, n_epochs=500, learning_rate=1e-4, weight_decay=1e-4, verbose=True):
#     """
#     Trains the DeepSurv model.

#     Parameters:
#         model: DeepSurv model instance.
#         train_data: Dictionary with keys 'x', 't', 'e' for training data.
#         valid_data: Optional. Dictionary with keys 'x', 't', 'e' for validation data.
#         n_epochs: Number of training epochs.
#         learning_rate: Learning rate for the optimizer.
#         weight_decay: L2 regularization coefficient.
#         verbose: Whether to print training progress.

#     Returns:
#         Dictionary containing training and validation metrics.
#     """
#     # Prepare data
#     x_train = torch.tensor(train_data['x'], dtype=torch.float32)
#     t_train = train_data['t']
#     e_train = torch.tensor(train_data['e'], dtype=torch.float32)

#     if valid_data:
#         x_valid = torch.tensor(valid_data['x'], dtype=torch.float32)
#         t_valid = valid_data['t']
#         e_valid = torch.tensor(valid_data['e'], dtype=torch.float32)

#     # Optimizer
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Training loop
#     metrics = {'train_loss': [], 'train_ci': [], 'valid_loss': [], 'valid_ci': []}
#     for epoch in range(n_epochs):
#         model.train()
#         optimizer.zero_grad()
#         loss = model.negative_log_likelihood(x_train, e_train)
#         loss.backward()
#         optimizer.step()

#         # Compute training metrics
#         metrics['train_loss'].append(loss.item())
#         train_ci = model.get_concordance_index(x_train, t_train, e_train.numpy())
#         metrics['train_ci'].append(train_ci)

#         # Validation metrics
#         if valid_data:
#             model.eval()
#             with torch.no_grad():
#                 valid_loss = model.negative_log_likelihood(x_valid, e_valid)
#                 metrics['valid_loss'].append(valid_loss.item())
#                 valid_ci = model.get_concordance_index(x_valid, t_valid, e_valid.numpy())
#                 metrics['valid_ci'].append(valid_ci)

#         # Print progress
#         if verbose and (epoch + 1) % 10 == 0:
#             print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}, Train CI: {train_ci:.4f}")

#     return metrics

# # # Example dataset
# # train_data = {
# #     'x': np.random.rand(100, 10),
# #     't': np.random.rand(100),
# #     'e': np.random.randint(0, 2, size=100)
# # }
# # valid_data = {
# #     'x': np.random.rand(50, 10),
# #     't': np.random.rand(50),
# #     'e': np.random.randint(0, 2, size=50)
# # }

# # # Initialize and train the model
# # model = DeepSurv(n_in=10, hidden_layers_sizes=[32, 16], activation="relu", dropout=0.5, batch_norm=True)
# # metrics = train_deepsurv(model, train_data, valid_data, n_epochs=100, learning_rate=1e-3)

# # # Print final metrics
# # print("Final Training Loss:", metrics['train_loss'][-1])
# # print("Final Training CI:", metrics['train_ci'][-1])


# # from torch_datasets import SimulatedData  # Ensure this is the correct import path

# # Initialize the SimulatedData object
# sim_data = SimulatedData(
#     N=1000,                  # Number of observations
#     hr_ratio=2.0,            # Hazard ratio
#     average_death=5,         # Average death time
#     censor_mode='end_time',  # Censoring mode ('end_time' or 'observed_p')
#     end_time=15,             # End time for censoring
#     num_features=10,         # Number of features
#     num_var=2,               # Number of variables affecting risk
#     treatment_group=True,    # Include treatment group
#     method='gaussian',       # Method for generating risk ('linear' or 'gaussian')
#     gaussian_config={        # Additional parameters for Gaussian risk
#         'c': 0.0,            # Offset for Gaussian function
#         'rad': 0.5           # Scale parameter for Gaussian function
#     },
#     as_tensor=True           # Store data as PyTorch tensors
# )

# # Access the generated dataset
# x = sim_data.x  # Covariates (features)
# t = sim_data.t  # Observed time events
# e = sim_data.e  # Censoring indicators (1 = observed, 0 = censored)
# hr = sim_data.hr  # True hazard ratios

# # Print some information about the dataset
# print("Covariates (x):", x.shape)
# print("Observed times (t):", t[:5])
# print("Censoring indicators (e):", e[:5])
# print("Hazard ratios (hr):", hr[:5])

# # Example: Use the dataset with PyTorch DataLoader
# from torch.utils.data import DataLoader

# # Create a DataLoader for batching
# dataloader = DataLoader(sim_data, batch_size=32, shuffle=True)

# # Iterate through the DataLoader
# for batch in dataloader:
#     print("Batch x shape:", batch['x'].shape)
#     print("Batch t:", batch['t'][:5])
#     print("Batch e:", batch['e'][:5])
#     break