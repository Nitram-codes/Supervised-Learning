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
# check if there are any missing entries in the data
print(df.isnull().sum())
# plot the distribution of the target variable using seaborn
# displot plots the data curve and the corresponding histogram
y = pd.Series(df['MEDV'], name = 'Median Home Value')
sns.set_style('darkgrid')
fig = plt.figure()
ax = sns.distplot(y, bins = 30)

# create a correlation matrix that measures the relationships between
# the variables
correlation_matrix = df.corr().round(2)
# use the seaborn heatmap function to visualise the matrix
# annot = True displays the numerical values on the heatmap
fig = plt.figure()
ax = sns.heatmap(correlation_matrix, annot = True, cmap = 'jet')
# plot some of the most important features with the target variable
ax = sns.pairplot(df[['MEDV', 'LSTAT', 'RM', 'PTRATIO']])

X = df.drop('MEDV', axis = 1)

# split into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42)
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

# the data must be shuffled
# earlier this was doen by train_test_split
df2 = shuffle(df)
X2 = df2.drop('MEDV', axis = 1)
y2 = pd.Series(df2['MEDV'], name = 'Median Home Value')

# carry out a cross validation of the unregularised model
unreg_cross_val = np.mean(cross_val_score(reg, X2, y2, cv = 5))
print('\n')
print('Cross Validation')
print('-------------------------------')
print('R^2 = {:.2f}'.format(unreg_cross_val))
# # make a prediction for a house with 8 rooms, a LSTAT of 10%, a PTRATIO 0f 18.
# # INDUS of 10 and TAX of 300
# price_prediction = reg.predict(np.array([8, 10, 18]).reshape(1, 3))

# perform regularised regression (ridge)
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
    ridge_cv_scores = cross_val_score(ridge, X2, y2, cv = 5)
    ridge_scores.append(np.mean(ridge_cv_scores))
fig4 = plt.figure()
ax3 = fig4.add_subplot(111)
ax3.set_xscale('log')
ax3.set_xlabel(r'Regularisation parameter, $\alpha$')
ax3.set_ylabel('$R^2$')
ax3.set_title('Linear Regression - Ridge')
ax3.plot(alpha_space, ridge_scores)

# from the plot it can be seen that there is no benefit to using ridge regression
# as optimum model performance occurs at alpha = 0
# create a ridge regressor with alpha = 0
ridge2 = Ridge(alpha = 0, normalize = True)
# fit to data
ridge2.fit(X_train, y_train)
# identify coefficients and intercept
coefficients = ridge2.coef_
intercept = ridge2.intercept_
# use cross_val method to return R^2 value
# this should be approximately equal to max(ridge_scores) and 'unreg_cross_val'
ridge2_cv_score = np.mean(cross_val_score(ridge2, X2, y2, cv = 5))
print('\n')
print('Regularised linear regression (ridge)')
print('Cross Validation')
print('-------------------------------')
print('R^2 = {:.2f}'.format(ridge2_cv_score))

# Perform Lasso regularised regression to see if model performance can
# be improved
alpha_space2 = np.linspace(0.0001, 0.5, 50)
lasso_scores = []
# create lasso instance
lasso = Lasso(normalize = True)
for alpha in alpha_space2:
    # set alpha
    lasso.alpha = alpha
    # use cross_val method to return R^2 value
    lasso_cv_scores = np.mean(cross_val_score(lasso, X2, y2, cv = 5))
    lasso_scores.append(lasso_cv_scores)
fig5 = plt.figure()
ax5 = fig5.add_subplot(111)
ax5.set_xlabel(r'Regularisation parameter, $\alpha$')
ax5.set_ylabel('$R^2$')
ax5.set_title('Linear Regression - Lasso')
ax5.set_xlim(0, 0.5)
ax5.set_ylim(0, 0.8)
ax5.plot(alpha_space2, lasso_scores)

print('\n')
print('Regularised linear regression (lasso)')
print('Cross Validation')
print('-------------------------------')
print('R^2 = {:.2f}'.format(max(lasso_scores)))
