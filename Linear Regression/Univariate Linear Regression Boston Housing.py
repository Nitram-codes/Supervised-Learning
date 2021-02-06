import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import seaborn as sns

boston = load_boston()
# boston is imported as a bunch, a subclass of dictionary type
# load as a DataFrame
df = pd.DataFrame(boston['data'], columns = boston['feature_names'])
# add the target data to df
df['MEDV'] = boston.target
# shuffle df
df = shuffle(df)
# check if there are any missing entries in the data
print(df.isnull().sum())

# we will focus on the feature LSTAT only (most highly correlated feature)
# reshape for scikit learn
X = df['LSTAT'].values.reshape(-1, 1)
y = df['MEDV'].values.reshape(-1, 1)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('% lower status of the population')
ax.set_ylabel("Median value of owner-occupied homes in $1000's")
ax.set_xlim(0, 40)
ax.set_ylim(0, 55)
ax.scatter(X, y, s = 8)

# split into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 18)

# create a regressor
reg = LinearRegression()
# fit the model to the training data
reg.fit(X_train, y_train)
# compute predictions for training data
y_train_pred = reg.predict(X_train)
# compute predictions for test data
y_test_pred = reg.predict(X_test)
print('Unregularised linear regression')
print('Training Set')
print('-------------------------------')
# compute the r squared value (for regression problems)
# this is the strength of the relationship between predicted y values and
# actual y values
# on this occasion how well model fits training data
# compares predictions for X_train (i.e. y_train_pred) with y_train
print('R^2 = {:.2f}'.format(reg.score(X_train, y_train)))
# compute root mean squared error
print('root mean squared error = {:.2f}'.format(np.sqrt(mean_squared_error(y_train, y_train_pred))))
print('\n')
print('Test set')
print('-------------------------------')
# here how well model fits test data
# compares predictions for X_test (i.e. y_test_pred) with y_test
print('R^2 = {:.2f}'.format(reg.score(X_test, y_test)))
print('root mean squared error = {:.2f}'.format(np.sqrt(mean_squared_error(y_test, y_test_pred))))

# see if ridge regression improves the model
# create a range of alphas to test and identify the optimum value
alpha_space = np.logspace(-5, 1, 50)
# lists to score stores
ridge_scores = []
# create a ridge regressor
ridge = Ridge(normalize = True)
# compute ridge scores (R^2 values) over a range of alphas
for alpha in alpha_space:
    # set alpha
    ridge.alpha = alpha
    # use cross_val method to return R^2 value
    ridge_cv_scores = cross_val_score(ridge, X, y, cv = 5)
    ridge_scores.append(np.mean(ridge_cv_scores))

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.set_title('Regularised linear regression (ridge) performance')
ax2.set_xlabel(r'Regularisation parameter, $\alpha$')
ax2.set_ylabel('$R^2$')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 0.6)
ax2.plot(alpha_space, ridge_scores)

# from the plot we see that the optimal alpha value is 0
# this is equal to no regularisation
ridge2 = Ridge(alpha = 0, normalize = True)
ridge2.fit(X_train, y_train)
print('\n')
print('Regularised linear regression')
print('Training Set')
print('-------------------------------')
print('R^2 = {:.2f}'.format(ridge2.score(X_train, y_train)))
print('\n')
print('Test set')
print('-------------------------------')
print('R^2 = {:.2f}'.format(ridge2.score(X_test, y_test)))

# plot both regression lines
x = np.linspace(0, 35, 100).reshape(-1, 1)
y2 = reg.predict(x)
y3 = ridge2.predict(x)
ax.plot(x, y2, color =  'red', label = 'unregularised', alpha = 0.5)
ax.plot(x, y3, color = 'green', label = 'regularised', alpha = 0.5)
ax.legend()
plt.show()