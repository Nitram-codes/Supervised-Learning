import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.utils import shuffle
import seaborn as sns

df = pd.read_csv('Auto.csv')
# shuffle data
df = shuffle(df)
# check if there are any missing values in the data
print(df.isnull().sum())
# assign the target variable and reshape for scikit learn
y = pd.Series(df['mpg'], name = 'miles per gallon')
# create a distribution plot of the data
sns.set_style('darkgrid')
fig = plt.figure()
ax = sns.distplot(y, bins = 30)

# create a correlation matrix that measures the relationships between
# the variables
correlation_matrix = df.corr().round(2)
# visualise the matrix
fig = plt.figure()
ax = sns.heatmap(correlation_matrix, annot = True, cmap = 'jet')
# plot pairplot of all features
# ax = sns.pairplot(df)

# there is strong multicollinearity among several features of the data
# model performance should be better with regression
X = df.drop(['mpg', 'name', 'horsepower'], axis = 1)

# create a regressor
reg = LinearRegression()
# perform cross validation on the unregularised model
unreg_cross_val = np.mean(cross_val_score(reg, X, y, cv = 5))
print('Unregularised linear regression')
print('--------------------------------')
print('Cross validation: {:.3f}'.format(unreg_cross_val))

# ElasticNet regularisation is a linear combination of Lasso and
# Risge regression, E = a*L1 + b*L2
# in scikit_learn this term is represented by the 'l1_ratio'
# an l1_ratio of 1 corresponds to Lasso regression
# a ratio less than 1 represents a linear combination L1 and L2

# create the L1 hyperparameter space
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}
# instantiate the ElasticNet regressor
elastic_net = ElasticNet()
# generate the GridSearch object
# we can use GridSearchCV to identify the optimum l1 value for the model
# GridSearchCV searches through a pre-defined list of l1 values (param_grid)
# and undertakes a cross-validation process to determine the best one
reg_cross_val = GridSearchCV(elastic_net, param_grid, cv = 5)
# fit to the data
reg_cross_val.fit(X, y)
# plot the results
test_scores = reg_cross_val.cv_results_['mean_test_score']
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.set_xlim(0, 1)
ax2.set_xlabel('L1 ratio')
ax2.set_ylabel('$R^2$')
ax2.plot(l1_space, test_scores)
# we can see that the R^2 score is highest when the L1 ratio is zero
# this corresponds to ridge regression

# perform regularised regression (ridge)
# create a range of alphas to test and identify the optimum value
alpha_space = np.logspace(-5, 1, 50)
# lists to score scores
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
fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.set_xscale('log')
ax3.set_xlabel(r'Regularisation parameter, $\alpha$')
ax3.set_ylabel('$R^2$')
ax3.plot(alpha_space, ridge_scores)
# we can see that the R^2 score is highest when alpha = 0
# this is unregularised regression
print('Regularised linear regression (ridge)')
print('--------------------------------------')
print('Cross validation: {:.3f}'.format(max(ridge_scores)))
