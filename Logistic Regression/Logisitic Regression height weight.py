import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('gender_height_weight.csv')

# convert height to cm
df['Height'] = df['Height']*2.54
# shuffle data entries
df = shuffle(df)

# generate histograms
fig = plt.figure()
ax = fig.add_subplot(121)
ax.set_xlabel('Height / cm)')
ax.set_ylabel('Count')
ax.hist(df['Height'], ec = 'k')
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Weight / lb')
ax2.set_ylabel('Count')
ax2.hist(df['Weight'], ec = 'k')

# generate dummy variables for 'Gender' column
gender_data = pd.get_dummies(df['Gender'], drop_first = True)
# add new column to the data frame
df = pd.concat([df, gender_data], axis = 1)

# compute and visualise correlation matrix of height and weight
X = df[['Height', 'Weight']]
corr_matrix = X.corr()
fig = plt.figure()
sns.heatmap(corr_matrix, annot = True)

# assign data and split into train and test sets
y = df['Male']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# create logistic regressor
log_reg = LogisticRegression(solver = 'liblinear')
# fit model to the training data
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
plot_data = df.drop('Male', axis = 1)
sns.set_style('darkgrid')
ax3 = sns.scatterplot(data = plot_data, x = 'Height', y = 'Weight', hue = 'Gender',
                      palette = ['red', 'blue'], alpha = 0.5)
ax3.set_xlabel('Height / cm')
ax3.set_ylabel('Weight / lb')

# retrieve coefficient and intercept data (a*x_1 + b*x_2 + c = 0)
a, b = log_reg.coef_[0][0], log_reg.coef_[0][1]
c = log_reg.intercept_[0]
# create x points for plot
x = np.linspace(135, 205, 100)
y_plot = -(a/b)*x - (c/b)
ax3.plot(x, y_plot, color = 'k', linestyle = 'dashed')

# predict classification of person 189cm tall and 200lb
# and a person 160cm tall and 137lb
new_candidates = {'Height': [189, 160], 'Weight': [200, 137]}
new_df = pd.DataFrame(new_candidates)
predictions = log_reg.predict(new_df)
