import numpy as np
import torch
from math import log, exp
from typing import Dict, Optional, Union

class SimulatedData:
    def __init__(self, hr_ratio: float,
                 average_death: float = 5,
                 censor_mode: str = 'end_time', 
                 end_time: Optional[float] = 15, 
                 observed_p: Optional[float] = None,
                 num_features: int = 10, 
                 num_var: int = 2,
                 treatment_group: bool = False):
        """
        Factory class for producing simulated survival data (PyTorch compatible version).
        
        Parameters:
            hr_ratio: lambda_max hazard ratio.
            average_death: average death time (mean of Exponential distribution).
            censor_mode: method to calculate censoring ['end_time', 'observed_p'].
            end_time: censoring time for 'end_time' mode.
            observed_p: percentage of observed deaths for 'observed_p' mode.
            num_features: size of observation vector.
            num_var: number of variables data depends on.
            treatment_group: whether to include binary treatment group covariate.
        """
        self.hr_ratio = hr_ratio
        self.censor_mode = censor_mode
        self.end_time = end_time
        self.observed_p = observed_p
        self.average_death = average_death
        self.treatment_group = treatment_group
        self.m = int(num_features) + int(treatment_group)
        self.num_var = num_var

    def _linear_H(self, x: np.ndarray) -> np.ndarray:
        """Linear risk function."""
        b = np.zeros((self.m,))
        b[0:self.num_var] = range(1, self.num_var + 1)
        return np.dot(x, b)

    def _gaussian_H(self, x: np.ndarray,
                   c: float = 0.0, 
                   rad: float = 0.5) -> np.ndarray:
        """Gaussian risk function."""
        max_hr, min_hr = log(self.hr_ratio), log(1.0 / self.hr_ratio)
        z = np.sum(np.square(x[:, 0:self.num_var] - c), axis=-1)
        return max_hr * (np.exp(-(z) / (2 * rad ** 2)))

    def generate_data(self, N: int,
                     method: str = 'gaussian', 
                     gaussian_config: Optional[Dict] = None,
                     **kwargs) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Generates survival data with optional PyTorch tensor output.
        
        Returns:
            Dictionary with:
                'x': (N,m) array of observations
                't': (N) array of observed times
                'e': (N) array of event indicators
                'hr': (N) array of true risk scores
        """
        if gaussian_config is None:
            gaussian_config = {}

        # Generate uniform patient data
        data = np.random.uniform(low=-1, high=1, size=(N, self.m))

        if self.treatment_group:
            data[:, -1] = np.random.randint(0, 2, size=(N,))

        # Calculate risk scores
        if method == 'linear':
            risk = self._linear_H(data)
        elif method == 'gaussian':
            risk = self._gaussian_H(data, **gaussian_config)
        else:
            raise ValueError(f"Unknown method: {method}")

        risk = risk - np.mean(risk)  # Center the hazard ratio

        # Generate survival times
        p_death = self.average_death * np.ones((N,))
        death_time = np.zeros((N,))
        
        for i in range(N):
            base_time = np.random.exponential(p_death[i])
            if self.treatment_group and data[i, -1] == 0:
                death_time[i] = base_time
            else:
                death_time[i] = base_time / exp(risk[i])

        # Handle censoring
        if self.censor_mode == 'observed_p':
            if self.observed_p is None:
                raise ValueError("observed_p required when censor_mode='observed_p'")
            end_time_idx = int(N * self.observed_p)
            self.end_time = np.sort(death_time.flatten())[end_time_idx]
        elif self.censor_mode != 'end_time':
            raise ValueError(f"Unknown censor_mode: {self.censor_mode}")

        censoring = np.ones((N,))
        death_time[death_time > self.end_time] = self.end_time
        censoring[death_time == self.end_time] = 0

        # Convert to PyTorch tensors if requested
        convert_to_tensor = kwargs.get('as_tensor', False)
        
        dataset = {
            'x': data.astype(np.float32),
            'e': censoring.astype(np.int32),
            't': death_time.astype(np.float32),
            'hr': risk.astype(np.float32)
        }

        if convert_to_tensor:
            dataset = {k: torch.from_numpy(v) for k, v in dataset.items()}

        return dataset
    

# # Generate NumPy data
sim = SimulatedData(hr_ratio=2.0, treatment_group=True)
data_np = sim.generate_data(1000, method='gaussian')

# Generate PyTorch tensor data
data_torch = sim.generate_data(1000, method='gaussian', as_tensor=True)

# Create PyTorch Dataset
from torch.utils.data import TensorDataset
dataset = TensorDataset(data_torch['x'], data_torch['t'], data_torch['e'])
dataset[0:2]
