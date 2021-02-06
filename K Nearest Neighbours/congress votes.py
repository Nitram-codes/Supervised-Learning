import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('congress_votes.csv')
data = data.replace('y', 1)
data = data.replace('n', 0)
# replace '?' with 0 for program to run
data = data.replace('?', 0)

# example of how to produce a bar chart of data of interest
plt.figure()
sns.countplot(x='education', hue='party', data=data, palette='RdBu')
plt.xticks([0,1], ['No', 'Yes'])
plt.show()

# create arrays for the the features (X) and the response variable (y)
# drop the 'party' row and use the values as the features X
X = data.drop('party', axis = 1).values
# y contains the outcomes and is the 'party' column
y = data['party'].values

# create a K-Nearest Neighbours classifier with 6 neighbours
knn = KNeighborsClassifier(n_neighbors = 6)
# fit the classifier to the data
knn.fit(X, y)
# generate some new X data to test prediction on
X_new = np.array([0.696, 0.286, 0.227, 0.551, 0.72, 0.423, 0.981, 0.685,\
                  0.481, 0.392, 0.343, 0.73, 0.44, 0.06, 0.4, 0.738]).reshape((1, 16))
prediction = knn.predict(X_new)
# democrat is predicted for the X_new data

# now the data will be split into training and test sets as is appropriate
# for real world data
# choose 20% of the data to be used for testing with 'test_size'
# 'random_state' allows the seeding of the random choice of training and test data
# stratify ensures the labels are distributed correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,\
                                                    random_state = 5, stratify = y)
knn1 = KNeighborsClassifier(n_neighbors = 8)
knn1.fit(X_train, y_train)
y_pred = knn1.predict(X_test)
# check the accuracy of the predicted data against y_test
print(knn1.score(X_test, y_test))


