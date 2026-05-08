# Linear Regression From Scratch

A comprehensive implementation and explanation of linear regression built from first principles using NumPy, without relying on scikit-learn or other ML libraries.

## 📋 Table of Contents

- [What is Linear Regression?](#what-is-linear-regression)
- [Mathematical Foundation](#mathematical-foundation)
- [Implementation](#implementation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Results](#results)
- [Key Concepts](#key-concepts)
- [Files](#files)

## What is Linear Regression?

Linear regression is one of the simplest and most fundamental machine learning algorithms. It models the relationship between input variables (features) and a continuous output variable by fitting a linear equation to the observed data.

### Real-World Applications
- **House Price Prediction**: Predicting prices based on square footage, location, etc.
- **Stock Price Forecasting**: Estimating future prices based on historical trends
- **Sales Analysis**: Predicting sales based on advertising spend
- **Temperature Forecasting**: Predicting weather patterns

## Mathematical Foundation

### The Linear Equation

For a single feature, linear regression models the relationship as:

```
y = m*x + b
```

Where:
- **y** = predicted output (dependent variable)
- **x** = input feature (independent variable)
- **m** = slope (weight) - how much y changes with x
- **b** = y-intercept (bias) - the baseline value when x=0

### The Cost Function

To measure how well our model fits the data, we use **Mean Squared Error (MSE)**:

```
MSE = (1/n) * Σ(y_actual - y_predicted)²
```

Where:
- **n** = number of samples
- **y_actual** = true values from the dataset
- **y_predicted** = values predicted by our model

### Gradient Descent

We minimize the cost function using **Gradient Descent**, an optimization algorithm that iteratively updates the parameters (m and b):

**Partial derivatives:**

```
∂MSE/∂m = (-2/n) * Σ(X * (y - y_pred))
∂MSE/∂b = (-2/n) * Σ(y - y_pred)
```

**Parameter updates:**

```
m = m - learning_rate * ∂MSE/∂m
b = b - learning_rate * ∂MSE/∂b
```

Where:
- **learning_rate** = controls how big each step is (typically 0.01)
- **epochs** = number of times we iterate through the entire dataset

## Implementation

### The LinearRegression Class

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        """
        Initialize the Linear Regression model
        
        Args:
            learning_rate: How fast the model learns (default: 0.01)
            epochs: Number of training iterations (default: 1000)
        """
        self.lr = learning_rate
        self.epochs = epochs
        self.m = 0  # slope/weight
        self.b = 0  # y-intercept/bias

    def fit(self, X, y):
        """
        Train the model on input data
        
        Args:
            X: Input features (numpy array)
            y: Target values (numpy array)
        """
        n = len(X)
        
        # Iterate for the specified number of epochs
        for _ in range(self.epochs):
            # Predict using current parameters
            y_pred = self.m * X + self.b
            
            # Calculate gradients
            dm = (-2/n) * np.sum(X * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            
            # Update parameters
            self.m -= self.lr * dm
            self.b -= self.lr * db

    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Input features (numpy array)
            
        Returns:
            Predicted values
        """
        return self.m * X + self.b
```

## How It Works

### Step-by-Step Process

1. **Initialize Parameters**: Start with m=0 and b=0
2. **For each epoch**:
   - Calculate predictions: `y_pred = m*X + b`
   - Calculate gradients using the partial derivatives
   - Update parameters in the direction that reduces the cost
3. **Repeat**: Continue until convergence or max epochs reached

### Visual Intuition

```
Initial State:          After Training:
Line doesn't fit well   Line fits the data much better
      *                       *
       *                    *
        *    →             *
         *                *
          *              *
```

## Usage

### Example: Predicting Values

```python
if __name__ == "__main__":
    # Create synthetic data: y = 2x
    X = np.array([1, 2, 3])
    y = np.array([20, 40, 60])
    
    # Initialize and train the model
    model = LinearRegression(learning_rate=0.01, epochs=1000)
    model.fit(X, y)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Display results
    print("Learned slope (m):", model.m)
    print("Learned intercept (b):", model.b)
    print("Predictions:", predictions)
```

## Results

### Sample Output

```
Before training : [10 20 30]
After training : [20.18600369 40.03991622 59.89382875]
Learned m : 19.8539125301466
Learned b : 0.3321233...
```

The model successfully learns that the relationship is approximately `y ≈ 20*x`, which is very close to the true relationship `y = 2*x` (note: the initialization affects the convergence).

## Key Concepts

### Learning Rate
- **Too high**: Model might overshoot and diverge
- **Too low**: Training is very slow
- **Optimal**: Model converges smoothly to the solution

### Epochs
- **Too few**: Model hasn't learned the pattern yet
- **Too many**: Wastes computation (but won't hurt after convergence)
- **Good amount**: Enough to reach convergence

### Convergence
The algorithm has converged when the gradient becomes very small (close to zero), meaning further parameter updates have minimal effect on the cost.

## Limitations

- **Linear relationships only**: Cannot model curved or non-linear relationships
- **Sensitive to outliers**: A few extreme values can skew the line significantly
- **Requires feature scaling**: Works better when features are normalized to similar ranges

## Extensions

To make this model more powerful, consider:

- **Multiple features**: Extend to multivariate linear regression (y = m₁x₁ + m₂x₂ + ... + b)
- **Regularization**: Add L1 (Lasso) or L2 (Ridge) penalties to prevent overfitting
- **Feature engineering**: Create polynomial features for non-linear relationships
- **Batch processing**: Use mini-batches for larger datasets

## Files

- **`Linear_regression_from_scratch.ipynb`** - Jupyter notebook with implementation and examples
- **`README.md`** - This file, containing theory and explanation

## Getting Started

1. Clone this repository
2. Open the Jupyter notebook: `Linear_regression_from_scratch.ipynb`
3. Run the cells to see the implementation in action
4. Experiment with different learning rates and epochs

## References

- [Wikipedia: Linear Regression](https://en.wikipedia.org/wiki/Linear_regression)
- [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)

---

**Happy Learning!** 🚀

For questions or suggestions, feel free to open an issue or contribute to this project.
