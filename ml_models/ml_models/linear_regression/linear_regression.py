from ml_models.base_model.base_model import BaseModel
from ml_models.utility.standard_scaler import StandardScaler
import numpy as np

class LinearRegression(BaseModel):
    
    def __init__(self, scale_X=False):
        self.scale_X = scale_X
        self.scaler = None



    def fit(self, X, y, use_gradient_descent=False, threshold=1e12, learning_rate=0.1, n_iter=1000):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        if use_gradient_descent:
            print("[INFO] Apply StandardScaler for gradient descent.")
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        if use_gradient_descent:
            self._fit_gradient_descent(X_b, y, learning_rate, n_iter)
        else:
            try:
                condition_number = np.linalg.cond(X_b.T @ X_b)
                if condition_number > threshold:
                    print("Matrix bad conditioned, using Gradient Descent")
                    self.scaler = StandardScaler()
                    X = self.scaler.fit_transform(X)
                    self._fit_gradient_descent(X_b, y, learning_rate, n_iter)
                else:
                    self.coef_ = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
            except np.linalg.LinAlgError:
                print("Matrix cannot be invertible, using Gradient Descent")
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
                self._fit_gradient_descent(X_b, y, learning_rate, n_iter)

    def _fit_gradient_descent(self, X_b, y, learning_rate=0.01, n_iter=100000):
        m = X_b.shape[0]  
        n = X_b.shape[1]  
        self.theta = np.zeros((n, 1)) 

        for iteration in range(n_iter):
            gradients = (2/m) * X_b.T @ (X_b @ self.theta - y)
            self.theta = self.theta - learning_rate * gradients

        self.coef_ = self.theta 


    def predict(self, X):
        X = np.array(X)
        if hasattr(self, 'scaler') and self.scaler is not None:
            X = self.scaler.transform(X)
        if not np.all(X[:, 0] == 1):
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.coef_

    def score(self, X, y):
        y = np.array(y).reshape(-1, 1)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
