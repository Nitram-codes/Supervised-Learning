from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import datasets
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

iris = datasets.load_iris()

# use only two features
X = iris['data'][:, :2]
y = iris['target']

# split in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# instantiate a logisitic regressor without regularisation
log_reg = LogisticRegression(C = 10000, multi_class = 'auto')
# fit to training data
log_reg.fit(X_train, y_train)
# predict the labels of the test data
y_pred = log_reg.predict(X_test)
# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# plot one vs all decision boundaries
fig = plt.figure()
sns.set_style('darkgrid')
sepal_length = X[:,0]
sepal_width = X[:,1]
names = {0:'setosa', 1:'versicolor', 2:'virginica'}
y_names = []
for i in y:
    y_names.append(names[i])
data = {'sepal length': sepal_length, 'sepal width': sepal_width, 'y': y,
        'Type': y_names}
df = pd.DataFrame(data)
ax = sns.scatterplot(data = df, x = 'sepal length', y = 'sepal width',
                     hue = 'Type')
# generate points for plotting decision boundaries
x = np.linspace(min(sepal_length), max(sepal_length), 100)
# get coefficients and intercepts for first boundary
a1, b1 = log_reg.coef_[0][0], log_reg.coef_[0][1]
c1 = log_reg.intercept_[0]
dec1 = -(a1/b1)*x - (c1/b1)
# get coefficients for second boundary
a2, b2 = log_reg.coef_[1][0], log_reg.coef_[1][1]
c2 = log_reg.intercept_[1]
dec2 = -(a2/b2)*x - (c2/b2)
# get coefficients for third boundary
a3, b3 = log_reg.coef_[2][0], log_reg.coef_[2][1]
c3 = log_reg.intercept_[2]
dec3 = -(a3/b3)*x - (c3/b3)
# plot boundaries
ax.plot(x, dec1, color = 'k', linestyle = 'dashed')
ax.plot(x, dec2, color = 'r', linestyle = 'dashed')
ax.plot(x, dec3, color = 'b', linestyle = 'dashed')
ax.set_xlim(4, 8)
ax.set_ylim(1.5, 5)
ax.set_title('One-vs-all decision boundaries', fontsize = 15)

# plot multiclass decision boundary
# create a meshgrid of points and then use the predict method to predict the
# classification (0, 1 or 2) of each point. A different colour is assigned to
# each of the three possible outcomes, generating the coloured areas on the plot
x_min = min(sepal_length) - 0.2
x_max = max(sepal_length) + 0.2
y_min = min(sepal_width) - 0.2
y_max = max(sepal_width) + 0.2
# step size in the grid
dr = 0.01
# generate meshgrid with x values from x_min to x_man and y values from
# y_min to y_max
X1, Y1 = np.meshgrid(np.arange(x_min, x_max, dr), np.arange(y_min, y_max, dr))
# transform meshgrid to a form compatible with sklearn
stack = np.column_stack((X1.ravel(), Y1.ravel()))
# make predictions on stack
predictions = log_reg.predict(stack)
# reshape predictions into grid for plotting
predictions = predictions.reshape(X1.shape)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.pcolormesh(X1, Y1, predictions, cmap= 'Pastel1')
# add data points to plot
setosa = df[df['Type'] == 'setosa']
versicolor = df[df['Type'] == 'versicolor']
virginica = df[df['Type'] == 'virginica']
ax2.scatter(setosa['sepal length'], setosa['sepal width'], color = 'blue',
            edgecolor = 'white', label = 'setosa')
ax2.scatter(versicolor['sepal length'], versicolor['sepal width'], color = 'orange',
            edgecolor = 'white', label = 'versicolor')
ax2.scatter(virginica['sepal length'], virginica['sepal width'], color = 'green',
            edgecolor = 'white', label = 'virginica')
ax2.set_xlabel('sepal length')
ax2.set_ylabel('sepal width')
ax2.set_title('Decision boundaries', fontsize = 15)
plt.legend()