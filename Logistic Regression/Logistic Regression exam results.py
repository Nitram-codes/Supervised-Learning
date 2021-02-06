import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Exam_marks.csv', header = None)
columns = ['Exam 1 mark', 'Exam 2 mark', 'Pass/Fail']
df.columns = columns
df = shuffle(df)

stats = df.describe()

# generate histograms
sns.set_style('whitegrid')
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_xlabel('Exam 1 mark')
ax.set_ylabel('Count')
ax.hist(df['Exam 1 mark'], ec = 'k')
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Exam 2 mark')
ax2.set_ylabel('Count')
ax2.hist(df['Exam 2 mark'], ec = 'k')

# compute and calculate the correlation matrix
X = df[['Exam 1 mark', 'Exam 2 mark']]
y = df['Pass/Fail']
corr_matrix = X.corr()
fig = plt.figure()
sns.heatmap(corr_matrix, annot = True)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# create logistic regressor
# in sklearn l2 regularisation is automatically applied to logistic regression
# C is the inverse of regularisation strength
# so to view the model without regularisation set C to high value
log_reg = LogisticRegression(solver = 'liblinear', C = 10000)
# fit to training data
log_reg.fit(X_train, y_train)
# predict the labels of the test data (y_test)
y_pred = log_reg.predict(X_test)

# compute and print the confusion matrix and classification report
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))
print('\n')
print('Classification report:')
print(classification_report(y_test, y_pred))

# plot data and decision boundary
fig = plt.figure()
ax2 = sns.scatterplot(data = df, x = 'Exam 1 mark', y = 'Exam 2 mark',
                      hue = 'Pass/Fail', style = 'Pass/Fail')

# retrieve coefficient and intercept data (a*x1 + b*x2 +c = 0)
a, b = log_reg.coef_[0][0], log_reg.coef_[0][1]
c = log_reg.intercept_[0]
# create x points for plot
x = np.linspace(30, 100, 100)
y_plot = -(a/b)*x - (c/b)
ax2.plot(x, y_plot, color = 'k', linestyle = 'dashed')

# make predictions for candidate who scores 80 and 60
# and a candidate who scores 45 and 55
new_candidates = {'Exam 1 mark': [80, 45], 'Exam 2 mark': [60, 55]}
new_df = pd.DataFrame(new_candidates)
predictions = log_reg.predict(new_df)