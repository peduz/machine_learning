# ml-models

## Description
`ml-models` is a Python library that implements various machine learning models, including regression, classification, and other fundamental techniques. The goal is to provide clear and modular implementations of common models, which can be easily extended and adapted to different use cases.

## Features
- **Linear Regression**: Implementation of a linear regression model, with support for training and prediction.
- **Gradient Descent**: An optimization algorithm for training models, including learning rate tuning.
- **Cost Functions**: Implementations of various common cost functions in machine learning.

## Installation
To install the library, use `poetry`:

```bash
poetry install
Usage
Here is an example of how to use the library:

python
Copy
Edit
from ml_models import LinearRegression

# Create a model
model = LinearRegression()

# Train the model with data
model.fit(X_train, y_train)

# Make predictions on new data
predictions = model.predict(X_test)
Contributing
If you'd like to contribute, please fork the project, create a branch, and submit a pull request. Make sure to include proper tests and documentation.

License
This project is licensed under the GNU license.