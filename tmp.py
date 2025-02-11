import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, 1 feature
y = 3 * X.squeeze() + np.random.randn(100) * 2  # True function: y = 3x + noise

# Fit OLS model
ols = LinearRegression()
ols.fit(X, y)

# Fit Ridge Regression with strong shrinkage
ridge = Ridge(alpha=10)
ridge.fit(X, y)

# Plot results
plt.scatter(X, y, color="gray", alpha=0.5, label="Data")
plt.plot(X, ols.predict(X), label="OLS (No Shrinkage)", color="blue")
plt.plot(X, ridge.predict(X), label="Ridge (Shrinkage)", color="red", linestyle="--")
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
plt.title("Effect of Shrinkage in Ridge Regression")
plt.show()
