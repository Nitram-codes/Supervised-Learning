import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv('poly_data.csv')
# assign variables
X = df.iloc[:, 1].values.reshape(-1, 1)
y = df.iloc[:, 2].values.reshape(-1, 1)

# attempt linear regression
reg = LinearRegression()
reg.fit(X, y)
# predict y
y_pred_0 = reg.predict(X)
# create x values for plot
x = np.linspace(0, 100, 100).reshape(-1, 1)
# predict y for plot
y_pred = reg.predict(x)
# get R^2 score
linear_score = reg.score(y, y_pred_0)

sns.set_style('darkgrid')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X, y, label = 'Original data')
ax.plot(x, y_pred, color = 'red', label = 'Linear regression')
ax.set_xlabel('Temperature')
ax.set_ylabel('Pressure')

# create a polynomial instance
poly = PolynomialFeatures(degree = 2)
# fit and transform the x vals
X_poly = poly.fit_transform(X)

# create linear regressor
reg2 = LinearRegression()
reg2.fit(X_poly, y)
# predict y
y_pred_2 = reg2.predict(X_poly)
# get R^2 score
poly_score = r2_score(y, y_pred_2)

# plot polynomial regression curve
poly_y = reg2.predict(poly.fit_transform(x))
ax.plot(x, poly_y, color = 'green', label = 'Polynomial regression')
ax.set_xlim(0, 100)
ax.set_ylim(-0.05,)
plt.legend()
plt.show()

print('Linear Regression R^2 = {:.3f}'.format(linear_score))
print('Polynomial Regression R^2 = {:.3f}'.format(poly_score))