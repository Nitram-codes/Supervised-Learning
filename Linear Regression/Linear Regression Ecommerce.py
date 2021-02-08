import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import seaborn as sns

df = pd.read_csv('ecommerce_data.csv', engine = 'python', error_bad_lines=False)
# shuffle data
df = shuffle((df))
# check for missing values in df
print(df.isnull().sum())

# create a correlation matrix that measures the relationships between
# the variables
correlation_matrix = df.corr().round(2)
# visualise the matrix
fig = plt.figure()
ax = sns.heatmap(correlation_matrix, annot = True, cmap = 'jet')
# plot pairplot of all features
ax = sns.pairplot(df)

# target variable is 'Yearly Amount Spent'
y = df['Yearly Amount Spent']
# select feature variables
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2,
                                                    random_state = 4)
# create the regressor
reg = LinearRegression()
# fit the mode to the training data
reg.fit(X_train, y_train)
# compute the predictions on the test data
y_pred = reg.predict(X_test)
# compute root mean squared error between 'y_test' and 'y_pred'
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# compute R^2
r_squared = reg.score(X_test, y_test)
print('Unregularised linear regression')
print('--------------------------------')
print('Training fit: R^2 = {:.3f}'.format(reg.score(X_train, y_train)))
print('Test fit: R^2 = {:.3f}, RMSE = {:.3f}'.format(r_squared, rmse))
print('\n')

# attempt ridge regression
# create a range of alphas to test and identify the optimum value
alpha_space = np.logspace(-5, 0, 50)
# lists to score stores
ridge_scores = []
ridge_training_scores = []
rmse2 = []
# create a ridge regressor
ridge = Ridge()
# compute ridge scores (R^2 values) over a range of alphas
for alpha in alpha_space:
    # set alpha
    ridge.alpha = alpha
    # fit data
    ridge.fit(X_train, y_train)
    # predict y
    y_pred = ridge.predict(X_test)
    # add RMSE to list
    rmse2.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    # add R^2 to list
    ridge_scores.append(ridge.score(X_test, y_test))
    # training scores
    ridge_training_scores.append(ridge.score(X_train, y_train))

# identify optimal regularisation parameter alpha
ridge_scores_array = np.asarray(ridge_scores)
optimal_alpha = alpha_space[np.argmax(ridge_scores_array)]
# identify corresponding training score and RMSE
rmse_score = rmse2[np.argmax(ridge_scores_array)]
ridge_training_scores_array = np.asarray(ridge_training_scores)
ridge_training_score = ridge_training_scores_array[np.argmax(ridge_scores_array)]

print('Ridge Regularised Linear Regression')
print('------------------------------------')
print('Training fit: R^2 = {:.3f}'.format(ridge_training_score))
print('Test fit: {:.3f}, RMSE = {:.3f}'.format(max(ridge_scores), rmse_score))
print('alpha = {:.3f}'.format(optimal_alpha))
print('\n')

# attempt lasso regression
alpha_space2 = np.logspace(-5, 0, 50)
lasso_scores = []
lasso_training_scores = []
rmse3 = []
# create lasso instance
lasso = Lasso()
for alpha in alpha_space2:
    # set alpha
    lasso.alpha = alpha
    # fit data
    lasso.fit(X_train, y_train)
    # predict y
    y_pred = lasso.predict(X_test)
    # add RMSE to list
    rmse3.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    # add R^2 to list
    lasso_scores.append(lasso.score(X_test, y_test))
    # add training scores to list
    lasso_training_scores.append(lasso.score(X_train, y_train))

# identify optimum alpha parameter
lasso_scores_array = np.asarray(lasso_scores)
optimal_alpha2 = alpha_space2[np.argmax(lasso_scores_array)]
# identify corresponding training score and RMSE
rmse_score2 = rmse3[np.argmax(lasso_scores_array)]
lasso_training_scores_array = np.asarray(lasso_training_scores)
lasso_training_score = lasso_training_scores_array[np.argmax(lasso_scores_array)]

print('Lasso Regularised Linear Regression')
print('------------------------------------')
print('Training fit: R^2 = {:.3f}'.format(lasso_training_score))
print('Test fit: {:.3f}, RMSE = {:.3f}'.format(max(lasso_scores), rmse_score2))
print('alpha = {:.3f}'.format(optimal_alpha2))
