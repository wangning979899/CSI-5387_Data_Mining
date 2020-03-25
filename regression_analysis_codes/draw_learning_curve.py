import numpy as np
import os
import pandas as pd
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC
from scipy.sparse import csr_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import os
os.chdir('./results')

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

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    plt.legend(loc="best")

    return plt

polynomial_features = PolynomialFeatures(degree=2,
                                            include_bias=False)
linear_regression = linear_model.LinearRegression()
estimator = Pipeline([("polynomial_features", polynomial_features),
                    ("linear_regression", linear_regression)])

title = r"Learning Curves (Linear Regression)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(estimator, title, data, label, ylim=(-5,2),
                    cv=cv, n_jobs=4)

plt.savefig("linear_regression_learning_curve.png")
# plt.show()