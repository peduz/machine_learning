from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in this library.
    """

    @abstractmethod
    def fit(self, X, y, use_gradient_descent=False, threshold=1e12):
        """
        Fit the model to the data.
        Parameters:
            X: features (e.g. numpy array or pandas DataFrame)
            y: target values (e.g. numpy array or pandas Series)
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the trained model.
        Parameters:
            X: features
        Returns:
            Predicted values
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """
        Evaluate the model on the given data.
        """
        pass
