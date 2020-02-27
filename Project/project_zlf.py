import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

df = pd.read_excel("Real estate valuation data set.xlsx")

# # data attributes:
# print(df.columns)

df = pd.DataFrame(df)

# # data demo
# print(df.iloc[0])

# remove the attribute identity number(No), this attribute is meaningless for this regression problem
df = df.drop(['No'],axis=1)

# handle missing values


# extract last column as label
df = np.array(df)
splitted_array = np.split(df,[-1],axis=1)
data = pd.DataFrame(splitted_array[0])
label = pd.DataFrame(splitted_array[1])

# attribute scaler
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)


# linear regression model: assume the relationship between Y and X is linear
# multivariable regression: Y(x1,x2,x3) = w1x1 + w2x2 + w3x3 + w0
linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(data,label)
print(linear_regression_model.coef_)

# PCA analysis or other dimensionality reduction

# correlated features and plot them

# draw the plot of each feature with the price

# draw learning rate plot

# There are three main errors(metrics) used to evaluate models.
# 1.mean absolute error
# 2.mean squared error
# 3.R2 score


