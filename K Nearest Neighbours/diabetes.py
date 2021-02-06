import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes.csv')
X = df.drop('diabetes', axis = 1)
y = df['diabetes'].values
# generate testing and training data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state = 42)
# instantiate a k-NN classifier with 6 neighbours
knn = KNeighborsClassifier(n_neighbors = 6)
# fit to the training data
knn.fit(X_train, y_train)
# predict the labels of the test data
y_pred = knn.predict(X_test)
# generate the confusion matrix and the classification report
con_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(con_matrix)
print(class_report)