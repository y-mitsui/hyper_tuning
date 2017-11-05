from __future__ import print_function
from __future__ import division
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

class _Ucb:
    def __init__(self, estimator, standard_scaler, kappa):
        self.estimator = estimator
        self.kappa = kappa
        self.standard_scaler = standard_scaler
        
    def getScore(self, x):
        new_x = self.standard_scaler.transform(x)
        mean, std = self.estimator.predict(new_x, return_std=True)
        return mean + self.kappa * std

class GpUCB:
    def __init__(self, bounds, alpha=1e-1, confidence_iterval=0.9, warm_up=10):
        kernel = Matern(nu=2.)
        self.gp = GaussianProcessRegressor(kernel, normalize_y=True, alpha=alpha, n_restarts_optimizer=2)
        kappa = np.sqrt(-2 * np.log(confidence_iterval))
        self.standard_scaler = StandardScaler()
        self.ucb = _Ucb(self.gp, self.standard_scaler, kappa)
        self.bounds = np.array(bounds)
        self.X = []
        self.Y = []
        self.warm_up = warm_up
        
    def set(self, sample_X, sample_y):
        self.X.append(sample_X)
        self.Y.append(sample_y)
        
    def get(self):
        if len(self.X) < self.warm_up:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
        
        X = self.standard_scaler.fit_transform(self.X)
        self.gp.fit(X, self.Y)
        res = differential_evolution(lambda x: -self.ucb.getScore(x.reshape(1, -1)), self.bounds, tol=0.001, maxiter=400, popsize=100)
        return res.x


if __name__ == "__main__":
    def func(x):
        return -(x + 3.) ** 2
        
    params = ((-20, 20), )
    gp_ucb = GpUCB(params, alpha=1e-1, confidence_iterval=0.8)
    for i in range(100):
        x = gp_ucb.get()
        gp_ucb.set(x, func(x[0]))
        print(i, x)
    

