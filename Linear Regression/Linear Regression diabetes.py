import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.utils import shuffle
import seaborn as sns

df = pd.read_csv('health_data.csv', delimiter = ' ')
# shuffle data frame
df = shuffle(df)
# check for missing values
print(df.isnull().sum())

# generate a correlation matrix that measures the relationships between the
# features
corr_matrix = df.corr().round(2)
# visualise the matrix
ax = sns.heatmap(corr_matrix, annot = True, cmap = 'jet')
# plot pairplots of all features
# ax = sns.pairplot(df)

# assign the features
X = df.drop('y', axis = 1).values
y = df['y'].values

# create a linear regressor
reg = LinearRegression(normalize = True)
# cross validate model
unreg_cross_val = np.mean(cross_val_score(reg, X, y, cv = 5))
print('Unregularised linear regression')
print('--------------------------------')
print('Cross validation: {:.3f}'.format(unreg_cross_val))

# perform regularised regression (ridge)
# create a range of alphas to test and identify the optimum value
alpha_space = np.logspace(-5, 1, 50)
# lists to score scores
ridge_scores = []
# create a ridge regressor
ridge = Ridge(normalize = True)
# compute R^2 for the range of alpha scores
for alpha in alpha_space:
    # set alpha
    ridge.alpha = alpha
    # cross validate model
    cv_score = np.mean(cross_val_score(ridge, X, y, cv = 5))
    ridge_scores.append(cv_score)

fig = plt.figure()
ax3 = fig.add_subplot(111)
ax3.set_xscale('log')
ax3.set_xlabel(r'Regularisation parameter, $\alpha$')
ax3.set_ylabel('$R^2$')
ax3.plot(alpha_space, ridge_scores)

print('\n')
print('Regularised linear regression')
print('------------------------------')
print('Cross validation: {:.3f}'.format(max(ridge_scores)))
print('Regularisation parameter alpha: {:.3f}'.format(alpha_space[np.argmax(ridge_scores)]))