import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


class MachineLearningModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        return "Training complete"

    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        return {"mse": mse, "r2": r2}

    def save_model(self):
        return json.dumps(
            {
                "coefficients": self.model.coef_.tolist(),
                "intercept": self.model.intercept_.item(),
            }
        )

    def load_model(self, params):
        params = json.loads(params)
        self.model.coef_ = np.array(params["coefficients"])
        self.model.intercept_ = params["intercept"]


class DataManager:
    def __init__(self):
        self.X = np.array([])
        self.y = np.array([])

    def add_data(self, X, y):
        if len(self.X) == 0:
            self.X = X
            self.y = y
        else:
            self.X = np.vstack((self.X, X))
            self.y = np.concatenate((self.y, y))

    def get_data(self):
        return self.X, self.y

    def clear_data(self):
        self.X = np.array([])
        self.y = np.array([])


# Example data and labels (kept for compatibility)
def example_data():
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([2, 3, 4])
    return X, y
