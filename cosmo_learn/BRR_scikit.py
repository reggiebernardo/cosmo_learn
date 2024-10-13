import numpy as np
from sklearn.linear_model import BayesianRidge


class BRR_sk:
    def __init__(self, n_order, tol=1e-6, init=[1., 1e-3]):
        """
        Initialize Bayesian Ridge Regression model.

        Parameters:
        - n_order (int): The order of the polynomial features.
        - tol (float): Tolerance for stopping criteria.
        - init (list): Initial values for alpha and lambda.
        """
        self.n_order = n_order
        self.tol = tol
        self.init = init

    def train(self, x_train, y_train, yerr_train):
        """
        Train the Bayesian Ridge Regression model.
    
        Parameters:
        - x_train (array-like): Training input data.
        - y_train (array-like): Target values.
        - yerr_train (array-like): Uncertainty of the target values.
    
        Returns:
        - reg0 (BayesianRidge): Trained Bayesian Ridge Regression model.
        """
        X_train = np.vander(x_train, N=self.n_order + 1, increasing=True)
        sample_weights = 1 / yerr_train**2
        sample_weights /= np.sum(sample_weights)
        
        reg0 = BayesianRidge(tol=self.tol, fit_intercept=False, compute_score=False)
        reg0.set_params(alpha_init=self.init[0], lambda_init=self.init[1])
        reg0.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store the trained model in the brr attribute
        self.brr = reg0
        
        return reg0

    def predict(self, x_test):
        """
        Predict target values and uncertainties for test data.

        Parameters:
        - x_test (array-like): Test input data.

        Returns:
        - predictions (dict): Dictionary containing predicted values and uncertainties.
        """
        # Assuming x_train, y_train, yerr_train are attributes of the class
        X_test = np.vander(x_test, self.n_order + 1, increasing=True)
        ymean, ystd = self.brr.predict(X_test, return_std=True)
        return {'z': x_test, 'Y': ymean, 'varY': ystd**2}
