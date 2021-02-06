from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# load the digits dataset, which is of type Bunch, similar to dictionaries
digits = datasets.load_digits()
# Bunch keys
print(digits.keys())
# Description of dataset
print(digits.DESCR)
# shape of an image
print(digits.images.shape)
# shape of the data arrays (flattened image)
print(digits.data.shape)
# view the 1111 image in the dataset of 1797
# this happens to be the number 5, the image is  8 x 8 pixels
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# X data has the dicitonary key 'data'
X = digits['data']
# y data has the dictionary key 'target'
y = digits['target']

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 42, stratify = y)
# create the k-NN classifier with 7 nearest neighbours
knn = KNeighborsClassifier(n_neighbors = 7)
# fit to the training data
knn.fit(X_train, y_train)
# score the model on the test data
print(knn.score(X_test, y_test))

# Now we will plot a model complexity curve to indicate over and under fitting
# create arrays to store train and test accuracies
neighbours = np.arange(1, 9)
train_accuracy = np.empty(len(neighbours))
test_accuracy = np.empty(len(neighbours))

# trial with different values of k, from 1 to 8
for i, k in enumerate(neighbours):
    # create the classifier
    knn1 = KNeighborsClassifier(n_neighbors = k)
    # fit to training data
    knn1.fit(X_train, y_train)
    # compute accuracy on the training set and assign to 'train_accuracy'
    train_accuracy[i] = knn1.score(X_train, y_train)
    # compute accuracy on the test set and assign to 'test_accuracy'
    test_accuracy[i] = knn1.score(X_test, y_test)

# generate plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('k-NN: Varying Number of neighbours')
ax.plot(neighbours, test_accuracy, label = 'Testing Accuracy')
ax.plot(neighbours, train_accuracy, label = 'Training Accuracy')
ax.legend()
ax.set_xlabel('Number of Neighbours')
ax.set_ylabel('Accuracy')
plt.show()
