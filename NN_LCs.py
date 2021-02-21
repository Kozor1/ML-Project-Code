import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import tree, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
plt.rcParams["figure.figsize"] = (3.5,2.5)

random_seed = 17

ds = ds.make_classification(n_samples=2000, n_features=5, n_informative=2, random_state=random_seed)

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

max_its = []
train_losses = []
test_losses = []

for iters in range(1,201):

    opt_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=[3,2], solver='lbfgs', random_state=random_seed, alpha=0.1, max_iter=iters)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        opt_clf.fit(X_train, y_train)

    max_its.append(iters)
    train_losses.append(accuracy_score(y_train, opt_clf.predict(X_train)))
    test_losses.append(accuracy_score(y_test, opt_clf.predict(X_test)))

plt.plot(max_its, train_losses, label = "Train Scores")
plt.plot(max_its, test_losses, label = "Test Scores")
plt.title("Accuracy vs Iterations")
plt.legend()
plt.show()
