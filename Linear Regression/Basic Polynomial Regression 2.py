import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('poly_data_salaries.csv')
# assign data and reshape for sklearn
X = df['Level'].values.reshape(-1, 1)
y = df['Salary'].values.reshape(-1, 1)

# attempt linear regression
reg = LinearRegression()
reg.fit(X, y)
# predict y
y_pred = reg.predict(X)
# get R^2 score
lin_score = reg.score(y, y_pred)

# generate x values for plot
x = np.linspace(min(X), max(X), 100)
# predict y for these x values
y_lin_plot = reg.predict(x)

sns.set_style('darkgrid')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('Position Level')
ax.set_ylabel('Salary')
ax.set_ylim(0, max(y) + 0.1*max(y))
ax.scatter(X, y, label = 'Original data')
ax.plot(x, y_lin_plot, color = 'g', label = 'Linear regression')

# create a polynomial instance
poly = PolynomialFeatures(degree = 2)
# fit and transform the features X
X_poly = poly.fit_transform(X)

# create linear regressor
reg2 = LinearRegression()
# fit to polynomial features
reg2.fit(X_poly, y)
# predict y
y_pred2 = reg2.predict(X_poly)
# get R^2 score
poly_score = r2_score(y, y_pred2)

# plot polynomial regression curve
y_poly_plot = reg2.predict(poly.fit_transform(x))
ax.plot(x, y_poly_plot, color = 'r', label = 'Polynomial regression')
plt.legend()
plt.show()

print('Linear Regression R^2 = {:.3f}'.format(lin_score))
print('Polynomial Regression R^2 = {:.3f}'.format(poly_score))
