import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt

df = pd.read_csv('Gapminder.csv')

# fertility data will be our single feature X
X = df['fertility'].values
# life expectancy is our outcome y
y = df['life'].values
# X and y must be reshaped for use with scikit-learn on this occasion
# we go from a 1D array with 139 elements to a 2D array with 139 rows and 1 column
X = X.reshape((-1, 1))
y = y.reshape((-1, 1))
# seaborn heatmap of correlations between the features
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,
                                                    random_state = 42)
# create the regressor
reg = LinearRegression()
# fit the model to the training data
reg.fit(X_train, y_train)
# compute the predictions on the test data
y_pred = reg.predict(X_test)
# compute the r squared value (for regression problems)
# this is the strength of the relationship between predicted y values and
# actual y values
# compares predictions for X_test (i.e. y_pred) with y_test
r_squared = reg.score(X_test, y_test)
# compute root mean squared error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# compute 5-fold cross_validation scores
# data is split into 5 sets. The model is trained on 4 sets whilst 1 is reserved
# for testing. The process is repeated until each set has had a turn being
# tested. The R^2 of each trial is returned
cv_scores = cross_val_score(reg, X, y, cv = 5)
print('R^2: {:.2f}'.format(r_squared))
print('Root mean squared error: {:.2f}'.format(rmse))
print('CV scores: {}'.format(cv_scores))
print('Average 5-fold CV score: {}'.format(np.mean(cv_scores)))

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(X, y)
ax.plot(X_test, y_pred, color = 'k')
ax.set_ylabel('Life Expectancy / years')
ax.set_xlabel('Fertility / number of children')

# Undertake Lasso regularisation on the data to identify the most important
# features of the data
# instantiate a Lasso regressor
# alpha is the regularisation parameter
# 'normalize' arguments ensures all features are on the same scale
lasso = Lasso(alpha = 0.4, normalize = True)
# fit to data
X1 = df.drop(['Region', 'life'], axis = 1)
lasso.fit(X1.values, y)
#compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
# child mortality turns out to be the most important feature in relation
# to life expectancy

# Undertake ridge regularisation of the data
# create a range of alphas to test and identify the optimum value
alpha_space = np.logspace(-4, 0, 50)
# lists to score stores
ridge_scores = []
ridge_scores_std = []
# create a ridge regressor
ridge = Ridge(normalize = True)
# compute ridge scores (R^2 values) over a range of alphas
for alpha in alpha_space:
    # set alpha
    ridge.alpha = alpha
    # use cross_val method to return R^2 value
    ridge_cv_scores = cross_val_score(ridge, X1, y, cv = 10)
    ridge_scores.append(np.mean(ridge_cv_scores))
    ridge_scores_std.append(np.std(ridge_cv_scores))
# plot the R^2 values (ridge_scores) against alpha to identify the optimum
# alpha value for the model
ax2 = fig.add_subplot(122)
ax2.plot(alpha_space, ridge_scores, color = 'r')
ax2.set_xscale('log')
ax2.set_xlabel('Alpha')
ax2.set_ylabel('CV Score +/- Std Error')
std_error = ridge_scores_std / np.sqrt(10)
# creates a fill effect of the standard error of the R^2 values
ax2.fill_between(alpha_space, ridge_scores + std_error, ridge_scores - std_error, alpha=0.2)

# ElasticNet regularisation is a linear combination of Lasso and
# Risge regression, E = a*L1 + b*L2
# in scikit_learn this term is represented by the 'l1_ratio'
# an l1_ratio of 1 corresponds to Lasso regression
# a ratio less than 1 represents a linear combination L1 and L2

# create the L1 hyperparameter space
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
# Instantiate the ElasticNet regressor
elastic_net = ElasticNet()
# generate the GridSearch object
# we can use GridSearchCV to identify the optimum l1 value for the model
# GridSearchCV searches through a pre-defined list of l1 values (param_grid)
# and undertakes a cross-validation process to determine the best one
gm_cv = GridSearchCV(elastic_net, param_grid, cv = 5)
# fit to the training data
gm_cv.fit(X_train, y_train)
# predict on the test set
y_pred2 = gm_cv.predict(X_test)
# r_squared compares predictions of X_test (y_pred2) with y_test
r_squared2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_pred2, y_test)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r_squared2))
print("Tuned ElasticNet MSE: {}".format(mse))