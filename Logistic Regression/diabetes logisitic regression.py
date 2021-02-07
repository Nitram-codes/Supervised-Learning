import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
X = df.drop('diabetes', axis = 1)
# scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y = df['diabetes'].values
# generate testing and training data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3,
                                   random_state = 42)
# create the logistic regression classifier
log_reg = LogisticRegression(solver='liblinear')
# fit to the data
log_reg.fit(X_train, y_train)
# predict the labels of the test data
y_pred = log_reg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# compute predicted y probabilities from X_test
# the first column is the probability that y = 0 and the second
# column is the probability that y = 1
# in this case we desire the second column
y_pred_prob = log_reg.predict_proba(X_test)[:,1]
# fpr = false positive rates
# tpr = true positive rates
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# plot Receiver Operating Characteristic (ROC) curve
# to identify optimum threshold in the model (default = 0.5)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr, tpr)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
plt.show()

# compute and print the area under the curve score
auc_score = roc_auc_score(y_test, y_pred)
cross_val_auc = cross_val_score(log_reg, X, y, cv = 5, scoring = 'roc_auc')
print('AUC: {}'.format(auc_score))
print('AUC scores computed using 5-fold cross-validation: {}'.format(cross_val_auc))

# for logistic regression regularisation C = 1/lambda
# we can use GridSearchCV to identify the optimum C value for the model
# GridSearchCV searches through a pre-defined list of C values (param_grid)
# and undertakes a cross-validation process to determine the best one

# setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}
# instantiate the GridSearch object
log_reg_cv = GridSearchCV(log_reg, param_grid, cv = 5)
# fit to data
log_reg_cv.fit(X, y)
print('Tuned Logistic Regression Parameters: {}'.format(log_reg_cv.best_params_))
print('Best Score is: {}'.format(log_reg_cv.best_score_))

# we will do the same process again but include an extra parameter
# in param_grid to identify whether L1 (Lasso) or L2 (Ridge) regression
# provides the best fit to the data
# it is best practice to use GridSearchCV on the training data only
# and keep some of the data separate for testing only
param_grid2 = {'C': c_space, 'penalty': ['l1', 'l2']}
log_reg_cv2 = GridSearchCV(log_reg, param_grid2, cv = 5)
log_reg_cv2.fit(X_train, y_train)
print('Tuned Logistic Regression Parameter: {}'.format(log_reg_cv2.best_params_))
print('Tuned Logistic Regression Accuracy: {}'.format(log_reg_cv2.best_score_))