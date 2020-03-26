import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import kernel_ridge
from sklearn import tree
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import svm
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
from regression_evaluation_metric import *
import os
path_current = os.getcwd()
print(path_current)
# os.chdir('./results')

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

def possitive_prediction_correction(prediction):
    for i in range(len(prediction)):
        if(prediction[i][0]<0):
            prediction[i][0] = 0
    return prediction

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
        f_linear.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
        f_linear.write("\n max_error: "+ str(max_error(prediction,y_test)))
        f_linear.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
        f_linear.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
        f_linear.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
        f_linear.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
        prediction = possitive_prediction_correction(prediction)
        f_linear.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
        f_linear.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
        f_linear.write("\n")
    f_linear.close()

def linear_regression_model_with_interaction():
    degrees = [1, 2, 3, 4, 5, 6]
    f_linear = open("linear_regression_multi_parameters_results.txt", 'w+')
    bias_list = [True, False]
    interaction_only_list = [True, False]
    for bias in bias_list:
        for interaction in interaction_only_list:
            for i in range(len(degrees)):
                polynomial_features = PolynomialFeatures(degree=degrees[i],
                                                        include_bias=bias,
                                                        interaction_only=interaction)
                linear_regression = linear_model.LinearRegression()
                pipeline = Pipeline([("polynomial_features", polynomial_features),
                                    ("linear_regression", linear_regression)])
                pipeline.fit(X_train,y_train)
                prediction = pipeline.predict(X_test)
                f_linear.write("\n LinearRegression with degree "+str(degrees[i])+", bias "+str(bias)+ ", interaction "+str(interaction)+" R2 score: "+str(pipeline.score(X_test,y_test)))
                f_linear.write("\n LinearRegression with degree "+str(degrees[i])+", bias "+str(bias)+ ", interaction "+str(interaction)+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
                f_linear.write("\n LinearRegression with degree "+str(degrees[i])+", bias "+str(bias)+ ", interaction "+str(interaction)+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
                f_linear.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
                f_linear.write("\n max_error: "+ str(max_error(prediction,y_test)))
                f_linear.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
                f_linear.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
                f_linear.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
                f_linear.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
                prediction = possitive_prediction_correction(prediction)
                f_linear.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
                f_linear.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
                f_linear.write("\n")
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

# Linear regression is highly sensitive to random errors, producing a large variance if features are dependent.
# This is the problem of multicollinearity.
# min_w||Xw-y||_2^2 -- L2 norm

# Ridge regression addresses the problem of multicollinearity by imposing a penalty on the size of coefficients.
# min_w||Xw-y||_2^2 + å||w||_2^2
# greater å, coefficients become more robust to collinearity

def Ridge_regression_model():
    f_ridge = open("ridge_regression_results.txt", 'w+')
    alpha_list = [0.005,0.05,0.5,5,50]
    for i in alpha_list:
        ridge_regression = linear_model.Ridge(alpha=i)
        ridge_regression.fit(X_train,y_train)
        prediction = ridge_regression.predict(X_test)
        f_ridge.write("\n RidgeRegression with alpha "+str(i)+" R2 score: "+str(ridge_regression.score(X_test,y_test)))
        f_ridge.write("\n RidgeRegression with alpha "+str(i)+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
        f_ridge.write("\n RidgeRegression with alpha "+str(i)+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
        f_ridge.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
        f_ridge.write("\n max_error: "+ str(max_error(prediction,y_test)))
        f_ridge.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
        f_ridge.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
        f_ridge.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
        f_ridge.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
        f_ridge.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
        f_ridge.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
        f_ridge.write("\n")
    f_ridge.close()

# 10-fold cross validation Ridge regression
# use grid-search GridSearchCV over a parameter grid to do model selection
def Ridge_cross_validation_model():
    f_ridge = open("Ridge_cross_validation_model_results.txt", 'w+')
    alpha_list = [0.005,0.05,0.5,5,50]
    ridge_regression = linear_model.RidgeCV(cv=10,alphas=np.array(alpha_list))
    ridge_regression.fit(X_train,y_train)
    prediction = ridge_regression.predict(X_test)
    f_ridge.write("\n 10-fold cross validation RidgeRegression R2 score: "+str(ridge_regression.score(X_test,y_test)))
    f_ridge.write("\n 10-fold cross validation RidgeRegression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_ridge.write("\n 10-fold cross validation RidgeRegression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_ridge.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_ridge.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_ridge.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_ridge.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_ridge.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_ridge.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_ridge.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_ridge.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_ridge.write("\n")
    f_ridge.close()

# Lasso uses coordinate descent as the algorithm to fit the coefficients
# min_x 1/(2n_samples) ||Xw - y||_2^2 + å||w||_1 ,L1 norm
# alpha parameter controls the degree of sparsity of the estimated coefficients
# as the lasso regression yields sparse models, it can thus be used to perform feature selection
def Lasso_regression():
    f_lasso = open("lasso_regression_results.txt", 'w+')
    alpha_list = [0.00005,0.0001,0.0005,0.001,0.005,0.05,0.5]
    for i in alpha_list:
        lasso_regression = linear_model.Lasso(alpha=i)
        lasso_regression.fit(X_train,y_train)
        prediction = lasso_regression.predict(X_test)
        prediction = np.reshape(prediction,(-1,1))
        f_lasso.write("\n LassoRegression with alpha "+str(i)+" R2 score: "+str(lasso_regression.score(X_test,y_test)))
        f_lasso.write("\n LassoRegression with alpha "+str(i)+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
        f_lasso.write("\n LassoRegression with alpha "+str(i)+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
        f_lasso.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
        f_lasso.write("\n max_error: "+ str(max_error(prediction,y_test)))
        f_lasso.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
        f_lasso.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
        f_lasso.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
        f_lasso.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
        f_lasso.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
        f_lasso.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
        f_lasso.write("\n")
    f_lasso.close()

# scitkit-learn exposes objects that set the Lasso alpha parameter by cross-validation:

def Lasso_cross_validation_model():
    f_lasso = open("lasso_cross_validation_model_results.txt", 'w+')
    alpha_list = [0.00005,0.0001,0.0005,0.001,0.005,0.05,0.5]
    lasso_regression = linear_model.LassoCV(cv=10,alphas=alpha_list)
    lasso_regression.fit(X_train,y_train)
    prediction = lasso_regression.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_lasso.write("\n 10-fold cross validation LassoRegression R2 score: "+str(lasso_regression.score(X_test,y_test)))
    f_lasso.write("\n 10-fold cross validation LassoRegression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_lasso.write("\n 10-fold cross validation LassoRegression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_lasso.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_lasso.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_lasso.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n")
    f_lasso.close()

# AIC is the Akaike information criterion
# BIC is the Bayes information criterion
# Such criteria are useful to select the value of the regularization parameter by making a trade-off between the goodness of fit and the complexity of the model

def Lasso_AIC():
    f_lasso = open("lasso_AIC_results.txt", 'w+')
    lasso_regression = linear_model.LassoLarsIC(criterion='aic',eps=0.002)
    lasso_regression.fit(X_train,y_train)
    prediction = lasso_regression.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_lasso.write("\n Lasso regression with Akaike information criterion R2 score: "+str(lasso_regression.score(X_test,y_test)))
    f_lasso.write("\n Lasso regression with Akaike information criterion Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_lasso.write("\n Lasso regression with Akaike information criterion Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_lasso.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_lasso.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_lasso.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n")
    f_lasso.close()

def Lasso_BIC():
    f_lasso = open("lasso_BIC_results.txt", 'w+')
    lasso_regression = linear_model.LassoLarsIC(criterion='bic',eps=0.002)
    lasso_regression.fit(X_train,y_train)
    prediction = lasso_regression.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_lasso.write("\n Lasso regression with Bayes information criterion R2 score: "+str(lasso_regression.score(X_test,y_test)))
    f_lasso.write("\n Lasso regression with Bayes information criterion Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_lasso.write("\n Lasso regression with Bayes information criterion Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_lasso.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_lasso.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_lasso.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_lasso.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_lasso.write("\n")
    f_lasso.close()

# Elastic-Net trade-off Lasso(sparse model, L1 norm) and Ridge(regularization, L2 norm)
# ElasticNetCV can be used to set the parameters alpha(å) and l1_ratio(p,rho) by cross-validation
# min_w (1/2*n_samples) ||Xw-y||_2^2 + åp||w||_1 + (å(1-p)/2)||w||_2^2
# Initially, some elastic nets do not meet the convergence because default tolerance for the optimization is too small although iteration number is large enough
def Elastic_net_regression():
    f_elastic = open("elastic_net_results.txt", 'w+')
    alpha_list = np.array([0.00005,0.0001,0.0005,0.001,0.005,0.05,0.5,5,50])
    rho_list = np.array([0.0,0.1,0.3,0.5,0.7,0.9,1.0])
    elastic_net = linear_model.ElasticNetCV(cv=10,l1_ratio=rho_list,alphas=alpha_list,max_iter=10000,tol=0.0001)
    elastic_net.fit(X_train,y_train)
    prediction = elastic_net.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_elastic.write("\n 10-fold cross validation Elastic Net R2 score: "+str(elastic_net.score(X_test,y_test)))
    f_elastic.write("\n 10-fold cross validation Elastic Net Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_elastic.write("\n 10-fold cross validation Elastic Net Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_elastic.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_elastic.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_elastic.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_elastic.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_elastic.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_elastic.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_elastic.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_elastic.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_elastic.write("\n")
    f_elastic.close()

# Least-angle regression(LARS) is a regression algorithm for high-dimensional data
# LARS is similar to forward stepwise regression
# at each step, it finds the feature most correlated with the target
# It is numberically efficient in contexts where the number of features is significanly greater than the number of samples.
# Since LARS is based upon an iterative refitting of the residuals, it would appear to be especially sensitive to the effects of noise.
# LassoLars yield the exact solution, which is a piecewise linear as a function of the norm of its coefficients.
# alpha -- penalty term
def least_angle_regression():
    f_lars = open("least_angle_regression_results.txt", 'w+')
    alpha_list = [0.0005,0.005,0.05,0.1]
    for i in alpha_list:
        LARS_regression = linear_model.LassoLars(alpha=i)
        LARS_regression.fit(X_train,y_train)
        prediction = LARS_regression.predict(X_test)
        prediction = np.reshape(prediction,(-1,1))
        f_lars.write("\n Least Angle Regression with alpha "+str(i)+" R2 score: "+str(LARS_regression.score(X_test,y_test)))
        f_lars.write("\n Least Angle Regression with alpha "+str(i)+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
        f_lars.write("\n Least Angle Regression with alpha "+str(i)+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
        f_lars.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
        f_lars.write("\n max_error: "+ str(max_error(prediction,y_test)))
        f_lars.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
        f_lars.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
        f_lars.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
        f_lars.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
        f_lars.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
        f_lars.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
        f_lars.write("\n")
    f_lars.close()

# Beyesian ridge regression is more robust to ill-posed problems(noise)
def Bayesian_ridgt_regression():
    f_bayes = open("Bayesian_ridge_regression_results.txt", 'w+')
    Bayesian_regression = linear_model.BayesianRidge()
    Bayesian_regression.fit(X_train,y_train)
    prediction = Bayesian_regression.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_bayes.write("\n Bayesian Ridge Regression R2 score: "+str(Bayesian_regression.score(X_test,y_test)))
    f_bayes.write("\n Bayesian Ridge Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_bayes.write("\n Bayesian Ridge Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_bayes.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_bayes.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_bayes.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_bayes.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_bayes.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_bayes.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_bayes.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_bayes.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_bayes.write("\n")
    f_bayes.close()

# Automatic Relevance Determination, similar to Bayesian Ridge Regression, lead to sparser coefficients w.
def ARD_regression():
    f_ARD = open("ARD_regression_results.txt", 'w+')
    ARD_regression_model = linear_model.ARDRegression()
    ARD_regression_model.fit(X_train,y_train)
    prediction = ARD_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_ARD.write("\n ARD Regression R2 score: "+str(ARD_regression_model.score(X_test,y_test)))
    f_ARD.write("\n ARD Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_ARD.write("\n ARD Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_ARD.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_ARD.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_ARD.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_ARD.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_ARD.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_ARD.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_ARD.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_ARD.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_ARD.write("\n")
    f_ARD.close()

# Stochastic Gradient Descent
# SGD default loss is squared_loss, same to ordinart least squares
# if loss is huber, same to Huber loss for robust regression
# if loss is epsilon_insensitive, same to support vector regression
def SGD_regression():
    f_SGD = open("SGD_regression_results.txt", 'w+')
    SGD_regression_model = linear_model.SGDRegressor()
    SGD_regression_model.fit(X_train,y_train)
    prediction = SGD_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_SGD.write("\n SGD Regression R2 score: "+str(SGD_regression_model.score(X_test,y_test)))
    f_SGD.write("\n SGD Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_SGD.write("\n SGD Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_SGD.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_SGD.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_SGD.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_SGD.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_SGD.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_SGD.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_SGD.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_SGD.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_SGD.write("\n")
    f_SGD.close()

# Passive aggressive algorithms, for large-scale learning, similar to the Perceptron in that they do not require a learning rate
def passive_aggressive_regression():
    f_passive_aggressive = open("passive_aggressive_regression_results.txt", 'w+')
    passive_aggressive_regression_model = linear_model.PassiveAggressiveRegressor()
    passive_aggressive_regression_model.fit(X_train,y_train)
    prediction = passive_aggressive_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_passive_aggressive.write("\n Passive Aggressive Regression R2 score: "+str(passive_aggressive_regression_model.score(X_test,y_test)))
    f_passive_aggressive.write("\n Passive Aggressive Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_passive_aggressive.write("\n Passive Aggressive Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_passive_aggressive.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_passive_aggressive.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_passive_aggressive.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_passive_aggressive.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_passive_aggressive.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_passive_aggressive.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_passive_aggressive.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_passive_aggressive.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_passive_aggressive.write("\n")
    f_passive_aggressive.close()
    
# Robust regression aims to fit a regression model in the presence of corrput data: either outliers in the model
# in general, robust fitting in high-dimensional setting(large n-features) is very hard
# both Theil Sen and RANSAC are unlikely to be as robust as HuberRegressor for the default parameters.
# RANSAC will deal better with large outliers in the y direction (most common situation).
# Theil Sen will cope better with medium-size outliers in the X direction, but this property will disappear in high-dimensional settings.
def robust_regreesion(): #RANdom SAmple Consensus
    f_robustness = open("robustness_regression_results.txt", 'w+')
    robustness_regression_model = linear_model.RANSACRegressor()
    robustness_regression_model.fit(X_train,y_train)
    prediction = robustness_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_robustness.write("\n Robustness Regression R2 score: "+str(robustness_regression_model.score(X_test,y_test)))
    f_robustness.write("\n Robustness Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_robustness.write("\n Robustness Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_robustness.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_robustness.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_robustness.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_robustness.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_robustness.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    prediction = possitive_prediction_correction(prediction)
    f_robustness.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n")
    f_robustness.close()

# RANSAC is good for strong outliers in the y direction
# TheilSen is good for small outliers, both in direction X and y, but has a break point above which it performs worse than OLS.
# The scores of HuberRegressor may not be compared directly to both TheilSen and RANSAC because it does not attempt to completely filter the outliers but lessen their effect.

# TheilsenRegressor estimator uses a generalization of the median in multiple dimensions. It is thus robust to multivariate outliers.
# performs no better than ordinary least squares in high dimension
# Theil-Sen is a median-based estimator

def Theil_Sen_robust_regreesion():
    f_robustness = open("Theil_Sen_robustness_regression_results.txt", 'w+')
    robustness_regression_model = linear_model.TheilSenRegressor()
    robustness_regression_model.fit(X_train,y_train)
    prediction = robustness_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_robustness.write("\n Theil Sen Robustness Regression R2 score: "+str(robustness_regression_model.score(X_test,y_test)))
    f_robustness.write("\n Theil Sen Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_robustness.write("\n Theil Sen Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_robustness.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_robustness.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_robustness.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_robustness.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_robustness.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    prediction = possitive_prediction_correction(prediction)
    f_robustness.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n")
    f_robustness.close()

# Huber regressor does not ignore the effect of the ouliers but gives a lesser weight to them
# Huber regressor should be more efficient to use on data with small number of samples
def huber_robust_regreesion():
    f_robustness = open("huber_robustness_regression_results.txt", 'w+')
    robustness_regression_model = linear_model.HuberRegressor()
    robustness_regression_model.fit(X_train,y_train)
    prediction = robustness_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_robustness.write("\n Huber Robustness Regression R2 score: "+str(robustness_regression_model.score(X_test,y_test)))
    f_robustness.write("\n Huber Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_robustness.write("\n Huber Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_robustness.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_robustness.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_robustness.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_robustness.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_robustness.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_robustness.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_robustness.write("\n")
    f_robustness.close()

# kernel ridge regression combines ridge regression and classification
# identical to support vector regression(SVR), KRR uses squared error loss
def kernel_ridge_regression():
    f_kernel_ridge = open("kernel_ridge_regression_results.txt", 'w+')
    kernel_ridge_regression_model = kernel_ridge.KernelRidge()
    kernel_ridge_regression_model.fit(X_train,y_train)
    prediction = kernel_ridge_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_kernel_ridge.write("\n Kernel Ridge Regression R2 score: "+str(kernel_ridge_regression_model.score(X_test,y_test)))
    f_kernel_ridge.write("\n Kernel Ridge Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_kernel_ridge.write("\n Kernel Ridge Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_kernel_ridge.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_kernel_ridge.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_kernel_ridge.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_kernel_ridge.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_kernel_ridge.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_kernel_ridge.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_kernel_ridge.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_kernel_ridge.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_kernel_ridge.write("\n")
    f_kernel_ridge.close()

# LinearSVR is same to the SVR with linear kernel
def support_vector_regression():
    f_SVR = open("support_vector_regression_results.txt",'w+')
    kernels_list = ['rbf','linear','poly','sigmoid'] # in this case, linear kernal is the best
    for kernel_name in kernels_list:
        support_vector_regression_model = svm.SVR(kernel=kernel_name)
        support_vector_regression_model.fit(X_train,y_train)
        prediction = support_vector_regression_model.predict(X_test)
        prediction = np.reshape(prediction,(-1,1))
        f_SVR.write("\n Support Vector Regression with kernel "+kernel_name+" R2 score: "+str(support_vector_regression_model.score(X_test,y_test)))
        f_SVR.write("\n Support Vector Regression with kernel "+kernel_name+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
        f_SVR.write("\n Support Vector Regression with kernel "+kernel_name+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
        f_SVR.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
        f_SVR.write("\n max_error: "+ str(max_error(prediction,y_test)))
        f_SVR.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
        f_SVR.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
        f_SVR.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
        f_SVR.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
        f_SVR.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
        f_SVR.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
        f_SVR.write("\n")
    support_vector_regression_model = svm.NuSVR(kernel='linear')
    support_vector_regression_model.fit(X_train,y_train)
    prediction = support_vector_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_SVR.write("\n Number Support Vector Regression with kernel linear R2 score: "+str(support_vector_regression_model.score(X_test,y_test)))
    f_SVR.write("\n Number Support Vector Regression with kernel linear Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_SVR.write("\n Number Support Vector Regression with kernel linear Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_SVR.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_SVR.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_SVR.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_SVR.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_SVR.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_SVR.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_SVR.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_SVR.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_SVR.write("\n")
    f_SVR.close()

# Neighbors-based regression can be used in cases where the data labels are continuous rather than discrete variables
def KNN_regression():
    f_KNN = open("k_nearest_neighbour_regression_results.txt",'w+')
    k_list = [1,3,5,7,9,11]
    weights_list = ['uniform', 'distance']
    for k in k_list:
        for weight in weights_list:
            KNN_regression_model = neighbors.KNeighborsRegressor(n_neighbors=k,weights=weight)
            KNN_regression_model.fit(X_train,y_train)
            prediction = KNN_regression_model.predict(X_test)
            prediction = np.reshape(prediction,(-1,1))
            f_KNN.write("\n KNN Regression with k "+str(k)+", weights "+weight+" R2 score: "+str(KNN_regression_model.score(X_test,y_test)))
            f_KNN.write("\n KNN Regression with k "+str(k)+", weights "+weight+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
            f_KNN.write("\n KNN Regression with k "+str(k)+", weights "+weight+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
            f_KNN.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
            f_KNN.write("\n max_error: "+ str(max_error(prediction,y_test)))
            f_KNN.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
            f_KNN.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
            f_KNN.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
            f_KNN.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
            f_KNN.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
            f_KNN.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
            f_KNN.write("\n")
    radius_list = [0.4, 0.6, 1.0, 10.0]
    for radius in radius_list:
        for weight in weights_list:
            KNN_regression_model = neighbors.RadiusNeighborsRegressor(radius=radius,weights=weight)
            KNN_regression_model.fit(X_train,y_train)
            prediction = KNN_regression_model.predict(X_test)
            prediction = np.reshape(prediction,(-1,1))
            f_KNN.write("\n Radius KNN Regression with radius "+str(radius)+", weights "+weight+" R2 score: "+str(KNN_regression_model.score(X_test,y_test)))
            f_KNN.write("\n Radius KNN Regression with radius "+str(radius)+", weights "+weight+" Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
            f_KNN.write("\n Radius KNN Regression with radius "+str(radius)+", weights "+weight+" Mean squared error: "+str(mean_squared_error(prediction,y_test)))
            f_KNN.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
            f_KNN.write("\n max_error: "+ str(max_error(prediction,y_test)))
            f_KNN.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
            f_KNN.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
            f_KNN.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
            f_KNN.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
            f_KNN.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
            f_KNN.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
            f_KNN.write("\n")
    f_KNN.close()

# Gaussian Process Regression
def gaussian_process_regression():
    f_gaussian_process = open("Gaussian_process_regression_results.txt", 'w+')
    gaussian_process_regression_model = gaussian_process.GaussianProcessRegressor()
    gaussian_process_regression_model.fit(X_train,y_train)
    prediction = gaussian_process_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_gaussian_process.write("\n Gaussian Process Regression R2 score: "+str(gaussian_process_regression_model.score(X_test,y_test)))
    f_gaussian_process.write("\n Gaussian Process Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_gaussian_process.write("\n Gaussian Process Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_gaussian_process.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_gaussian_process.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_gaussian_process.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_gaussian_process.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_gaussian_process.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_gaussian_process.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    prediction = possitive_prediction_correction(prediction)
    f_gaussian_process.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_gaussian_process.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_gaussian_process.write("\n")
    f_gaussian_process.close()

def decision_tree_regression():
    f_decision_tree = open("decision_tree_regression_results.txt", 'w+')
    decision_tree_regression_model = tree.DecisionTreeRegressor()
    decision_tree_regression_model.fit(X_train,y_train)
    prediction = decision_tree_regression_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_decision_tree.write("\n Decision Tree Regression R2 score: "+str(decision_tree_regression_model.score(X_test,y_test)))
    f_decision_tree.write("\n Decision Tree Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_decision_tree.write("\n Decision Tree Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_decision_tree.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_decision_tree.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_decision_tree.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_decision_tree.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_decision_tree.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_decision_tree.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_decision_tree.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_decision_tree.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_decision_tree.write("\n")
    f_decision_tree.close()

def voting_regression():
    f_voting = open("voting_regression_results.txt", 'w+')
    rg1 = linear_model.ARDRegression()
    rg2 = linear_model.HuberRegressor()
    rg3 = neighbors.RadiusNeighborsRegressor(radius=0.4,weights='distance')
    rg4 = linear_model.LassoLars(alpha=0.005)
    polynomial_features = PolynomialFeatures(degree=2,
                                                include_bias=False,interaction_only=False)
    linear_regression = linear_model.LinearRegression()
    rg5 = Pipeline([("polynomial_features", polynomial_features),
                            ("linear_regression", linear_regression)])
    rg6 = linear_model.LassoLarsIC(criterion='aic')
    ensemble_model = ensemble.VotingRegressor(estimators=[('ard',rg1),
                                                ('hubber',rg2),
                                                ('radiusKNN',rg3),
                                                ('lassolars',rg4),
                                                ('multipolynomial',rg5),
                                                ('lassolarsAIC',rg6)])
    y = np.array(y_train).ravel()
    ensemble_model.fit(X_train,y)
    prediction = ensemble_model.predict(X_test)
    prediction = np.reshape(prediction,(-1,1))
    f_voting.write("\n Voting Regression R2 score: "+str(ensemble_model.score(X_test,y_test)))
    f_voting.write("\n Voting Regression Mean absolute error: "+str(mean_absolute_error(prediction,y_test)))
    f_voting.write("\n Voting Regression Mean squared error: "+str(mean_squared_error(prediction,y_test)))
    f_voting.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_voting.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_voting.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_voting.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_voting.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_voting.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_voting.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_voting.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_voting.write("\n")
    f_voting.close()

# Isotonic regression fits a non-decreasing function to data
# In practice, this list of elements forms a function that is piecewise linear
# In isotonic regression, X should be 1d array

linear_regression_model()
linear_regression_model_with_interaction()
PCA_analysis()
Ridge_regression_model()
Ridge_cross_validation_model()
Lasso_regression()
Lasso_cross_validation_model()
Lasso_AIC()
Lasso_BIC()
Elastic_net_regression()
least_angle_regression()
Bayesian_ridgt_regression()
ARD_regression()
SGD_regression()
passive_aggressive_regression()
robust_regreesion()
Theil_Sen_robust_regreesion()
huber_robust_regreesion()
kernel_ridge_regression()
support_vector_regression()
KNN_regression()
gaussian_process_regression()
decision_tree_regression()
voting_regression()






