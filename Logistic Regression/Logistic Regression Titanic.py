import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score

df = pd.read_csv('titanic_passenger_list.csv')

# discard columns with a lot of missing entries
df = df[['pclass', 'survived', 'sex', 'age', 'fare', 'embarked']]
df = shuffle(df)
# drop rows with nan values
df = df.dropna()

# generate useful boxplots
sns.set_style('darkgrid')
sns.boxplot(df['pclass'], df['age'])
fig = plt.figure()
sns.boxplot(df['pclass'], df['fare'])

# generate useful histograms and countplots
fig = plt.figure()
plt.hist(df['age'])
fig = plt.figure()
sns.countplot(x = 'survived', data = df)
fig = plt.figure()
sns.countplot(x = 'survived', hue = 'sex', data = df)
fig = plt.figure()
sns.countplot(x = 'survived', hue = 'pclass', data = df)


# generate dummy variables for non-numeric columns
# drop_first = True ensures there is one column only and not
# corresponding male and female columns
sex_data = pd.get_dummies(df['sex'], drop_first = True)
embarked_data = pd.get_dummies(df['embarked'], drop_first = True)

# add new columns to the dataframe
df = pd.concat([df, sex_data, embarked_data], axis = 1)
# drop original columns
df = df.drop(['embarked', 'sex'], axis = 1)

# split data into X and y and training and testing sets
X = df.drop('survived', axis = 1)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# create the logistic regression classifier
log_reg = LogisticRegression(solver = 'liblinear')
# fit model on the training data
log_reg.fit(X_train, y_train)
# predict the labels of the test data
y_pred = log_reg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot Receiver Operating Characteristic (ROC) curve

# compute predicted y probabilities from X_test
# the first column is the probability that y = 0 and the second
# column is the probability that y = 1
# in this case we desire the second column
y_pred_prob = log_reg.predict_proba(X_test)[:,1]
# fpr = false positive rates
# tpr = true positive rates
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

fig2 = plt.figure()
ax = fig2.add_subplot(111)
ax.plot([0, 1], [0, 1], color  = 'k')
ax.plot(fpr, tpr)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
plt.show()

# Compute and print the Area Under the Curve (AUC) score
auc_score = roc_auc_score(y_test, y_pred)
cross_val_auc = cross_val_score(log_reg, X, y, cv = 5, scoring = 'roc_auc')
print('AUC: {}'.format(auc_score))
print('AUC mean score computed using 5-fold cross-validation: {}'.format(np.mean(cross_val_auc)))

# Regularisation does not improve model performance