from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris['data']
y = iris['target']

# split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# instantiate logistic regressor with no regularisation
log_reg = LogisticRegression(C = 10000, multi_class = 'auto')
# fit to training data
log_reg.fit(X_train, y_train)
# predict the labels of the test data
y_pred = log_reg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# make predictions
X_new = np.array([6.9, 3.2, 5.1, 1.9]).reshape(1, 4)
predictions = log_reg.predict(X_new)
