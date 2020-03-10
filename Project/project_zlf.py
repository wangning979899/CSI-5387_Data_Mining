import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

df = pd.read_excel("Real estate valuation data set.xlsx")

# # data attributes:
# print(df.columns)

df = pd.DataFrame(df)

# # data demo
# print(df.iloc[0])

# remove the attribute identity number(No), this attribute is meaningless for this regression problem
df = df.drop(['No'],axis=1)

# handle missing values
# no missing values in this dataset

# extract last column as label
df = np.array(df)
splitted_array = np.split(df,[-1],axis=1)
data = pd.DataFrame(splitted_array[0])
label = pd.DataFrame(splitted_array[1])

# attribute scaler
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)

# training and testing splitting
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=0)

# There are three main errors(metrics) used to evaluate models.
# 1.mean absolute error
# 2.mean squared error
# 3.R2 score
def mean_absolute_error(prediction,y_test):
    sum = 0
    n = len(prediction)
    for i in range(n):
        sum += abs(prediction[i][0]-y_test.iloc[i][0])
    return sum/n

def mean_squared_error(prediction,y_test):
    sum = 0
    n = len(prediction)
    for i in range(n):
        sum += (prediction[i][0]-y_test.iloc[i][0]) ** 2
    return sum/n

# linear regression model: assume the relationship between Y and X is linear
# multivariable regression: Y(x1,x2,x3) = w1x1 + w2x2 + w3x3 + w0
# Conclusion: degree 2 is the best one

def linear_regression_model():
    degrees = [1, 2, 3, 4, 5]
    f_linear = open("linear_regression_results.txt", 'w+')
    for i in range(len(degrees)):
        polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                include_bias=False)
        linear_regression = linear_model.LinearRegression()
        pipeline = Pipeline([("polynomial_features", polynomial_features),
                            ("linear_regression", linear_regression)])
        pipeline.fit(X_train,y_train)
        prediction = pipeline.predict(X_test)
        f_linear.write("\n LinearRegression with degree "+str(degrees[i])+" R2 score: "+str(pipeline.score(X_test,y_test)))
        f_linear.write("\n LinearRegression with degree "+str(degrees[i])+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
        f_linear.write("\n LinearRegression with degree "+str(degrees[i])+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_linear.close()

# some prediction examples

def PCA_analysis():
    # PCA analysis or other dimensionality reduction
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_train)
    # print(X_pca)
    polynomial_features = PolynomialFeatures(degree=2,include_bias=False)
    linear_regression = linear_model.LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                        ("linear_regression", linear_regression)])
    pipeline.fit(X_pca,y_train)
    prediction = pipeline.predict(X_pca)

    # Data for plotting
    plt.plot(X_pca, prediction,'b-')
    plt.plot(X_pca, y_train,'ro')
    plt.xlabel('features')
    plt.ylabel('label')
    plt.title('linear regression image with PCA')
    plt.savefig("linear_regression_image.png")
    # plt.show()

# correlated features and plot them

# draw the plot of each feature with the price

# draw learning curve plot




