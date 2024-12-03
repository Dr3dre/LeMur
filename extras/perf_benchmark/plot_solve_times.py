import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def plot_solve_times(csv_path, x_col, x_label, title):
    # Load the data
    data = pd.read_csv(csv_path)
    
    # Extract the data
    mean_times = data['mean']
    std_devs = data['std_dev']
    x = data[x_col]
    
    # Plot the data
    plt.errorbar(x, mean_times, yerr=std_devs, fmt='o', label='Mean solve time with std dev')
    plt.xlabel(x_label)
    plt.ylabel('Solve Time (s)')
    
    # Fit a polynomial regression model
    X = x.values.reshape(-1, 1)
    y = mean_times.values
    poly = PolynomialFeatures(degree=2)  # You can change the degree to fit higher-order polynomials
    X_poly = poly.fit_transform(X)
    # add exponential term to the model to fit the data better
    X_poly = np.concatenate((X_poly, np.exp(X)), axis=1)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Predict future values
    future_orders = np.arange(1, max(x)*1.1).reshape(-1, 1)
    future_orders_poly = poly.transform(future_orders)
    future_orders_poly = np.concatenate((future_orders_poly, np.exp(future_orders)), axis=1)
    future_times = model.predict(future_orders_poly)
    
    # y limit to 1.1 times the max value
    plt.ylim(0, 1.1*max(mean_times+std_devs))
    
    # Plot the regression line
    plt.plot(future_orders, future_times, label='Regression', linestyle='--')
    
    plt.legend()
    plt.title(title)
    plt.show()

# Usage
plot_solve_times('incremental_orders/results.csv', 'num_orders', 'Number of Orders (1000 Kg each)', 'Solve Times vs Number of Orders and KG')

plot_solve_times('incremental_horizon/results.csv', 'horizon', 'Horizon Days', 'Solve Times vs horizon days')