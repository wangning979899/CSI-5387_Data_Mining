# reference link: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
import math
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model,neighbors,ensemble
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import os
os.chdir('./results')

# Have outliers in the data and make sure they are outliers: use MAE, more robust to outliers
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

# The best possible score is 1.0, lower values are worse.
def explained_variance_score(prediction,y_test):
    return metrics.explained_variance_score(y_test,prediction)

# captures the worst case error
def max_error(prediction,y_test):
    return metrics.max_error(y_test,prediction)

# The median_absolute_error is particularly interesting because it is robust to outliers. 
# The loss is calculated by taking the median of all absolute differences between the target and the prediction.
def median_absolute_error(prediction,y_test):
    return metrics.median_absolute_error(y_test,prediction)

def root_mean_squared_error(prediction,y_test):
    sum = 0
    n = len(prediction)
    for i in range(n):
        sum += (prediction[i][0]-y_test.iloc[i][0]) ** 2
    return math.sqrt(sum/n)

# R2 score can be negative because because the model can be arbitrarily worse
def R_squared(prediction,y_test):
    return metrics.r2_score(y_test,prediction)

def mean_squared_percentage_error(prediction,y_test):
    return sum(np.reshape(np.array(((y_test-prediction)/y_test)**2),(-1)))/len(y_test)

def mean_absolute_percentage_error(prediction,y_test):
    return sum(np.reshape(np.array(abs((y_test-prediction)/y_test)),(-1)))/len(y_test)

def mean_squared_logarithmic_error(prediction,y_test):
    return metrics.mean_squared_log_error(y_test,prediction)

def root_mean_squared_logarithmic_error(prediction,y_test):
    return math.sqrt(metrics.mean_squared_log_error(y_test,prediction))

def evaluation_function(file_name,prediction,y_test):
    f_linear = open(file_name+"_results.txt", 'w+')
    f_linear.write("\n mean_absolute_error: "+ str(mean_absolute_error(prediction,y_test)))
    f_linear.write("\n mean_squared_error: "+ str(mean_squared_error(prediction,y_test)))
    f_linear.write("\n explained_variance_score: "+ str(explained_variance_score(prediction,y_test)))
    f_linear.write("\n max_error: "+ str(max_error(prediction,y_test)))
    f_linear.write("\n median_absolute_error: "+ str(median_absolute_error(prediction,y_test)))
    f_linear.write("\n root_mean_squared_error: "+ str(root_mean_squared_error(prediction,y_test)))
    f_linear.write("\n R_squared: "+ str(R_squared(prediction,y_test)))
    f_linear.write("\n mean_squared_percentage_error: "+ str(mean_squared_percentage_error(prediction,y_test)))
    f_linear.write("\n mean_absolute_percentage_error: "+ str(mean_absolute_percentage_error(prediction,y_test)))
    f_linear.write("\n mean_squared_logarithmic_error: "+ str(mean_squared_logarithmic_error(prediction,y_test)))
    f_linear.write("\n root_mean_squared_logarithmic_error: "+ str(root_mean_squared_logarithmic_error(prediction,y_test)))
    f_linear.close()

df = pd.read_excel("Real estate valuation data set.xlsx")
df = pd.DataFrame(df)
df = df.drop(['No'],axis=1)
df = np.array(df)
splitted_array = np.split(df,[-1],axis=1)
data = pd.DataFrame(splitted_array[0])
label = pd.DataFrame(splitted_array[1])
scaler = preprocessing.MinMaxScaler()
data = scaler.fit_transform(data)
data = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=0)

# def linear_regression_model():
#     polynomial_features = PolynomialFeatures(degree=2,
#                                             include_bias=False)
#     linear_regression = linear_model.LinearRegression()
#     pipeline = Pipeline([("polynomial_features", polynomial_features),
#                         ("linear_regression", linear_regression)])
#     pipeline.fit(X_train,y_train)
#     prediction = pipeline.predict(X_test)

#     evaluation_function("regression_evaluation_metric",prediction,y_test)

# def voting_regression():
#     rg1 = linear_model.ARDRegression()
#     rg2 = linear_model.HuberRegressor()
#     rg3 = neighbors.RadiusNeighborsRegressor(radius=0.4,weights='distance')
#     rg4 = linear_model.LassoLars(alpha=0.005)
#     polynomial_features = PolynomialFeatures(degree=2,
#                                                 include_bias=False,interaction_only=False)
#     linear_regression = linear_model.LinearRegression()
#     rg5 = Pipeline([("polynomial_features", polynomial_features),
#                             ("linear_regression", linear_regression)])
#     rg6 = linear_model.LassoLarsIC(criterion='aic')
#     ensemble_model = ensemble.VotingRegressor(estimators=[('ard',rg1),
#                                                 ('hubber',rg2),
#                                                 ('radiusKNN',rg3),
#                                                 ('lassolars',rg4),
#                                                 ('multipolynomial',rg5),
#                                                 ('lassolarsAIC',rg6)])
#     y = np.array(y_train).ravel()
#     ensemble_model.fit(X_train,y)
#     prediction = ensemble_model.predict(X_test)
#     prediction = np.reshape(prediction,(-1,1))
#     evaluation_function("voting_regression",prediction,y_test)

def ridge_path():
    # Compute paths

    n_alphas = 200
    alphas = np.logspace(0, 6, n_alphas)

    coefs = []
    for a in alphas:
        ridge = linear_model.Ridge(alpha=a, fit_intercept=False)
        ridge.fit(X_train, np.reshape(np.array(y_train),(-1)))
        coefs.append(ridge.coef_)

    # #############################################################################
    # Display results

    ax = plt.gca()

    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Ridge coefficients as a function of the regularization')
    plt.axis('tight')
    # plt.show()
    plt.savefig("ridge_path.png")

def plot_ic_criterion(model, name, color,EPSILON):
    alpha_ = model.alpha_ + EPSILON
    alphas_ = model.alphas_ + EPSILON
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')


def model_selection():
    # This is to avoid division by zero while doing np.log10
    EPSILON = 1e-5

    # #############################################################################
    # LassoLarsIC: least angle regression with BIC/AIC criterion

    model_bic = linear_model.LassoLarsIC(criterion='bic')
    model_bic.fit(data, label)
    # alpha_bic_ = model_bic.alpha_

    model_aic = linear_model.LassoLarsIC(criterion='aic')
    model_aic.fit(data, label)
    # alpha_aic_ = model_aic.alpha_

    plt.figure()
    plot_ic_criterion(model_aic, 'AIC', 'b', EPSILON)
    plot_ic_criterion(model_bic, 'BIC', 'r', EPSILON)
    plt.legend()
    plt.title('Information-criterion for model selection')
    plt.savefig('information_criterion_model_selection.png')

    # #############################################################################
    # LassoCV: coordinate descent

    # Compute paths
    model = linear_model.LassoCV(cv=10).fit(data, label)

    # Display results
    m_log_alphas = -np.log10(model.alphas_ + EPSILON)

    plt.figure()
    ymin, ymax = 20, 300
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_ + EPSILON), linestyle='--', color='k',
                label='alpha: CV estimate')

    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent ')
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.savefig('lasso_model_selection.png')

    # #############################################################################
    # LassoLarsCV: least angle regression

    # Compute paths
    model = linear_model.LassoLarsCV(cv=10).fit(data, label)

    # Display results
    m_log_alphas = -np.log10(model.cv_alphas_ + EPSILON)

    plt.figure()
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
            label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha CV')
    plt.legend()

    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars')
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.savefig('lasso_Lars_model_selection.png')
    # plt.show()


# linear_regression_model()
# voting_regression()
# ridge_path()
# model_selection()
