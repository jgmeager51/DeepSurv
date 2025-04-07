import numpy as np
import torch
from torch.utils.data import Dataset
from math import log, exp
from typing import Dict, Optional, Union


class SimulatedData(Dataset):
    def __init__(self,
                 N: int,
                 hr_ratio: float,
                 average_death: float = 5,
                 censor_mode: str = 'end_time',
                 end_time: Optional[float] = 15,
                 observed_p: Optional[float] = None,
                 num_features: int = 10,
                 num_var: int = 2,
                 treatment_group: bool = False,
                 method: str = 'gaussian',
                 gaussian_config: Optional[Dict] = None,
                 as_tensor: bool = True):
        """
        PyTorch-compatible dataset for simulated survival data.

        Parameters:
            N: Number of observations.
            hr_ratio: Hazard ratio (lambda_max).
            average_death: Average death time (mean of Exponential distribution).
            censor_mode: Method to calculate censoring ['end_time', 'observed_p'].
            end_time: Censoring time for 'end_time' mode.
            observed_p: Percentage of observed deaths for 'observed_p' mode.
            num_features: Number of features in the dataset.
            num_var: Number of variables affecting risk.
            treatment_group: Whether to include a binary treatment group covariate.
            method: Risk generation method ['linear', 'gaussian'].
            gaussian_config: Configuration for Gaussian risk generation.
            as_tensor: Whether to store data as PyTorch tensors.
        """
        self.hr_ratio = hr_ratio
        self.censor_mode = censor_mode
        self.end_time = end_time
        self.observed_p = observed_p
        self.average_death = average_death
        self.treatment_group = treatment_group
        self.m = int(num_features) + int(treatment_group)
        self.num_var = num_var
        self.method = method
        self.gaussian_config = gaussian_config or {}
        self.as_tensor = as_tensor

        # Generate the dataset
        self.dataset = self.generate_data(N)

        # Store data as tensors if requested
        if self.as_tensor:
            self.x = torch.from_numpy(self.dataset['x'])
            self.t = torch.from_numpy(self.dataset['t'])
            self.e = torch.from_numpy(self.dataset['e'])
            self.hr = torch.from_numpy(self.dataset['hr'])
        else:
            self.x = self.dataset['x']
            self.t = self.dataset['t']
            self.e = self.dataset['e']
            self.hr = self.dataset['hr']

    def _linear_H(self, x: np.ndarray) -> np.ndarray:
        """Linear risk function."""
        b = np.zeros((self.m,))
        b[0:self.num_var] = range(1, self.num_var + 1)
        return np.dot(x, b)

    def _gaussian_H(self, x: np.ndarray, c: float = 0.0, rad: float = 0.5) -> np.ndarray:
        """Gaussian risk function."""
        max_hr, min_hr = log(self.hr_ratio), log(1.0 / self.hr_ratio)
        z = np.sum(np.square(x[:, 0:self.num_var] - c), axis=-1)
        return max_hr * (np.exp(-(z) / (2 * rad ** 2)))

    def generate_data(self, N: int) -> Dict[str, np.ndarray]:
        """
        Generates survival data.

        Returns:
            Dictionary with:
                'x': (N, m) array of observations.
                't': (N,) array of observed times.
                'e': (N,) array of event indicators.
                'hr': (N,) array of true risk scores.
        """
        # Generate uniform patient data
        data = np.random.uniform(low=-1, high=1, size=(N, self.m))

        if self.treatment_group:
            data[:, -1] = np.random.randint(0, 2, size=(N,))

        # Calculate risk scores
        if self.method == 'linear':
            risk = self._linear_H(data)
        elif self.method == 'gaussian':
            risk = self._gaussian_H(data, **self.gaussian_config)
        else:
            raise ValueError(f"Unknown method: {self.method}")

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

        return {
            'x': data.astype(np.float32),
            't': death_time.astype(np.float32),
            'e': censoring.astype(np.int32),
            'hr': risk.astype(np.float32)
        }

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            't': self.t[idx],
            'e': self.e[idx],
            'hr': self.hr[idx]
        }
    


from torch.utils.data import DataLoader

# Create a dataset
dataset = SimulatedData(
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
    as_tensor=True
)

# Use with DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Check a batch
for batch in dataloader:
    print(batch['x'].shape, batch['t'][:5], batch['e'][:5])
    break

