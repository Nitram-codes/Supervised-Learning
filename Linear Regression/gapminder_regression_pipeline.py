import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

df = pd.read_csv('Gapminder.csv')
# assign the X and y data
X = df.drop(['life', 'Region'], axis = 1).values
y = df['life'].values

# setup the pipeline steps
# SimpleImputer will replace any 'NaN' values with the mean of the
# corresponding column
# StandardScaler will scale the X features
# ElasticNet is a regularised regression method that uses a linear
# combination of lasso and ridge regression
steps = [('imputation', SimpleImputer(missing_values = np.nan, strategy = 'mean')),
         ('scaler', StandardScaler()), ('elasticnet', ElasticNet())]
# create the pipeline
pipeline = Pipeline(steps)
# specify the hyperparameter space
# this will be the regression parameter to be optimised
# l1 ratio of 1 is solely lasso regression, anything else is a combination
parameters = {'elasticnet__l1_ratio': np.linspace(1, 1, 30)}
# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4,
                                                    random_state = 42)
# create the GridSearchCV object to find the optimum l1 ratio
gm_cv = GridSearchCV(pipeline, parameters, cv = 5)
# fit to the training set
gm_cv.fit(X_train, y_train)
# compute the R^2 metric
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))