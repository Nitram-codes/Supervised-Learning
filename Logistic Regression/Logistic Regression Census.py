import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore

# import data
df = pd.read_csv('census_train.txt', sep = ', ', header = None)
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gains', 'capital_loss', 'hours_per_week', 'native_country', 'income']
# assign column names
df.columns = cols
# shuffle data
df = shuffle(df)
# check if there are any missing entries in the data
print(df.isnull().sum())

# remove any rows containing '?'
for col in cols:
    # get indexes of column where value is '?'
    indexes = df[df[col] == '?'].index
    # drop all rows given by 'indexes'
    df.drop(indexes, inplace = True)

# drop irrelevant or unwanted columns
df = df.drop(['fnlwgt', 'education', 'native_country', 'relationship'], axis = 1)

# create countplots of observations
plt.figure()
sns.countplot(y = 'workclass', data = df, hue = 'income')
plt.figure()
sns.countplot(y = 'race', data = df, hue = 'income')
plt.figure()
sns.countplot(y = 'sex', data = df, hue = 'income')
plt.figure()
sns.countplot(y = 'occupation', data = df, hue = 'income')

# split into numerical and categorical data
num_data = df[['age', 'education_num', 'capital_gains', 'capital_loss',
               'hours_per_week']]
cat_data = df.drop(num_data.columns, axis = 1)
cat_data = cat_data.drop('income', axis = 1)

# scale the numerical data
# z = (x - mu)/s
# z = standard score, mu = column mean, s = column standard deviation
# x = training sample
num_data = num_data.apply(zscore)

# convert categorical data into dummy variable integers
cat_cols = cat_data.columns
for cat in cat_cols:
    # generate dummy variables for column
    new_col = pd.get_dummies(cat_data[cat], drop_first = True)
    # add new column to df
    cat_data = pd.concat([cat_data, new_col], axis = 1)
    # drop original column
    cat_data = cat_data.drop(cat, axis = 1)

# recombine numerical and categorical data
X = pd.concat([num_data, cat_data], axis = 1)
# assign the target variable y
y = pd.get_dummies(df['income'], drop_first = True)

# split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=21)

# find the optimal C value and penalty type
# create logistic regressor
log_reg = LogisticRegression(penalty = 'l2')
c_space = np.logspace(-1, 3, 50)
param_grid = {'C': c_space}
# perform gridsearch cross-validation
log_reg_cv = GridSearchCV(log_reg, param_grid, cv = 5)
log_reg_cv.fit(X_train, y_train)
print('Tuned Logistic Regression Parameter: {}'.format(log_reg_cv.best_params_))
print('Tuned Logistic Regression Accuracy: {}'.format(log_reg_cv.best_score_))

c = log_reg_cv.best_params_['C']

# create plot
sns.set_style('darkgrid')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.set_xlim(0,)
ax.set_ylim(0,)

# generate AUC plot for logistic regression with optimal C value
log_reg2 = LogisticRegression(penalty = 'l2', C = c)
log_reg2.fit(X_train, y_train)
# predict y_test
y_test_pred = log_reg2.predict(X_test)
# compute area under curve score
auc_score = roc_auc_score(y_test, y_test_pred)
# get probabilities that y_test is predicted 1
y_pred_prob = log_reg2.predict_proba(X_test)[:,1]
# get false positive and true positive rates
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plot ROC curve on ax
ax.plot(fpr, tpr, label = 'AUC = {:.2f}'.format(auc_score))
plt.legend()

# generate confusion matrix
print(confusion_matrix(y_test, y_test_pred))
# generate classification report
print(classification_report(y_test, y_test_pred))

# get coefficients and plot
coeffs = log_reg2.coef_[0]
plt.figure()
bar_names = X.columns
sns.barplot(bar_names, coeffs)
plt.xticks(rotation = 90)