from math import log, exp
import numpy as np

class SimulatedData:
    def __init__(self, hr_ratio,
        average_death = 5,
        censor_mode = 'end_time', end_time = 15, observed_p = None,
        num_features = 10, num_var = 2,
        treatment_group = False):
        """
        Factory class for producing simulated survival data.
        Current supports two forms of simulated data:
            Linear:
                Where risk is a linear combination of an observation's features
            Nonlinear (Gaussian):
                A gaussian combination of covariates

        Parameters:
            hr_ratio: lambda_max hazard ratio.
            average_death: average death time that is the mean of the
                Exponentional distribution.
            censor_mode: the method to calculate whether a patient is censored.
                Options: ['end_time', 'observed_p']
                'end_time': requires the parameter end_time, which is used to censor any patient with death_time > end_time
                'observed_p': requires the parammeter observed_p, which is the percentage of patients with observed death times
            end_time: censoring time that represents an 'end of study'. Any death
                time greater than end_time will be censored.
            num_features: size of observation vector. Default: 10.
            num_var: number of varaibles simulated data depends on. Default: 2.
            treatment_group: True or False. Include an additional covariate
                representing a binary treatment group.
        """

        self.hr_ratio = hr_ratio
        self.censor_mode = censor_mode
        self.end_time = end_time
        self.observed_p = observed_p
        self.average_death = average_death
        self.treatment_group = treatment_group
        self.m = int(num_features) + int(treatment_group)
        self.num_var = num_var

    def _linear_H(self,x):
        """
        Calculates a linear combination of x's features.
        Coefficients are 1, 2, ..., self.num_var, 0,..0]

        Parameters:
            x: (n,m) numpy array of observations

        Returns:
            risk: the calculated linear risk for a set of data x
        """
        # Make the coefficients [1,2,...,num_var,0,..0]
        b = np.zeros((self.m,))
        b[0:self.num_var] = range(1,self.num_var + 1)

        # Linear Combinations of Coefficients and Covariates
        risk = np.dot(x, b)
        return risk

    def _gaussian_H(self,x,
        c= 0.0, rad= 0.5):
        """
        Calculates the Gaussian function of a subset of x's features.

        Parameters:
            x: (n, m) numpy array of observations.
            c: offset of Gaussian function. Default: 0.0.
            r: Gaussian scale parameter. Default: 0.5.

        Returns:
            risk: the calculated Gaussian risk for a set of data x
        """
        max_hr, min_hr = log(self.hr_ratio), log(1.0 / self.hr_ratio)

        # Z = ( (x_0 - c)^2 + (x_1 - c)^2 + ... + (x_{num_var} - c)^2)
        z = np.square((x - c))
        z = np.sum(z[:,0:self.num_var], axis = -1)

        # Compute Gaussian
        risk = max_hr * (np.exp(-(z) / (2 * rad ** 2)))
        return risk

    def generate_data(self, N,
        method = 'gaussian', gaussian_config = {},
        **kwargs):
        """
        Generates a set of observations according to an exponentional Cox model.

        Parameters:
            N: the number of observations.
            method: the type of simulated data. 'linear' or 'gaussian'.
            guassian_config: dictionary of additional parameters for gaussian
                simulation.

        Returns:
            dataset: a dictionary object with the following keys:
                'x' : (N,m) numpy array of observations.
                't' : (N) numpy array of observed time events.
                'e' : (N) numpy array of observed time intervals.
                'hr': (N) numpy array of observed true risk.

        See:
        Peter C Austin. Generating survival times to simulate cox proportional
        hazards models with time-varying covariates. Statistics in medicine,
        31(29):3946-3958, 2012.
        """

        # Patient Baseline information
        data = np.random.uniform(low= -1, high= 1,
            size = (N,self.m))

        if self.treatment_group:
            data[:,-1] = np.squeeze(np.random.randint(0,2,(N,1)))
            print(data[:,-1])

        # Each patient has a uniform death probability
        p_death = self.average_death * np.ones((N,1))

        # Patients Hazard Model
        # \lambda(t|X) = \lambda_0(t) exp(H(x))
        #
        # risk = True log hazard ratio
        # log(\lambda(t|X) / \lambda_0(t)) = H(x)
        if method == 'linear':
            risk = self._linear_H(data)

        elif method == 'gaussian':
            risk = self._gaussian_H(data,**gaussian_config)

        # Center the hazard ratio so population dies at the same rate
        # independent of control group (makes the problem easier)
        risk = risk - np.mean(risk)

        # Generate time of death for each patient
        # currently exponential random variable
        death_time = np.zeros((N,1))
        for i in range(N):
            if self.treatment_group and data[i,-1] == 0:
                death_time[i] = np.random.exponential(p_death[i])
            else:
                death_time[i] = np.random.exponential(p_death[i]) / exp(risk[i])

        # If Censor_mode is 'observed_p': then find the end time in which observed_p percent of patients have an observed death
        if self.censor_mode is 'observed_p':
            if self.observed_p is None:
                raise ValueError("Parameter observed_p must be porivded if censor_mode is configured to 'observed_p'")
            end_time_idx = int(N * self.observed_p)
            self.end_time = np.sort(death_time.flatten())[end_time_idx]

        # Censor anything that is past end time
        censoring = np.ones((N,1))
        death_time[death_time > self.end_time] = self.end_time
        censoring[death_time == self.end_time] = 0

        # Flatten Arrays to Vectors
        death_time = np.squeeze(death_time)
        censoring = np.squeeze(censoring)

        dataset = {
            'x' : data.astype(np.float32),
            'e' : censoring.astype(np.int32),
            't' : death_time.astype(np.float32),
            'hr' : risk.astype(np.float32)
        }

        return dataset



# from deepsurv.datasets import SimulatedData

# Initialize the SimulatedData object
sim_data = SimulatedData(
    hr_ratio=2.0,            # Hazard ratio
    average_death=5,         # Average death time
    censor_mode='end_time',  # Censoring mode ('end_time' or 'observed_p')
    end_time=15,             # End time for censoring
    num_features=10,         # Number of features
    num_var=2,               # Number of variables affecting risk
    treatment_group=True     # Include treatment group
)

# Generate a dataset
dataset = sim_data.generate_data(
    N=1000,                  # Number of observations
    method='gaussian',       # Method for generating risk ('linear' or 'gaussian')
    gaussian_config={        # Additional parameters for Gaussian risk
        'c': 0.0,            # Offset for Gaussian function
        'rad': 0.5           # Scale parameter for Gaussian function
    }
)
type(dataset)
# Access the generated dataset
x = dataset['x']  # Covariates (features)
t = dataset['t']  # Observed time events
e = dataset['e']  # Censoring indicators (1 = observed, 0 = censored)
hr = dataset['hr']  # True hazard ratios

# Print some information about the dataset
print("Covariates (x):", x.shape)
print("Observed times (t):", t[:5])
print("Censoring indicators (e):", e[:5])
print("Hazard ratios (hr):", hr[:5])

print("Type of dataset:", type(dataset))