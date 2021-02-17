import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import tree, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
plt.rcParams["figure.figsize"] = (3.5,2.5)

random_seed = 17

ds = ds.make_classification(n_samples=2000, n_features=5, n_informative=2, random_state=random_seed)

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# First gridsearch
param_dict = {
    'solver': ['lbfgs', 'sgd', 'adam'],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
}

clf = MLPClassifier(max_iter=5000, random_state=random_seed)
grid_object = GridSearchCV(estimator = clf, param_grid = param_dict, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_object.fit(X_train, y_train)
best_params = grid_object.best_params_
print(best_params)

opt_clf = MLPClassifier(activation='logistic', alpha=0.1, solver='lbfgs', random_state=random_seed)

# Learning Curve Plots
train_sizes, train_scores, validation_scores = learning_curve(opt_clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)
# LC Plot
plt.plot(train_sizes, av_train_scores, label='train scores')
plt.plot(train_sizes, av_validation_scores, label ='validation scores')
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Scores")
plt.ylim([0.90, 1.0])
plt.legend()
plt.show()

# Validation Curves
hidden_layer_one = np.arange(1,11)
hidden_layer_two = np.arange(1,11)
param_range = [[i,j] for i in hidden_layer_one for j in hidden_layer_two]

train_scores, validation_scores = validation_curve(opt_clf, X_train, y_train, 'hidden_layer_sizes', param_range, cv=10, n_jobs=-1)
av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)
# VC Table
print(param_range)
print(av_train_scores)
print(av_validation_scores)

hidden_layer_one = np.arange(1,251)
param_range = [[i,3] for i in hidden_layer_one]

hidden_layer_two = np.arange(1,251,10)
param_range = [[2,j] for j in hidden_layer_two]

train_scores, validation_scores = validation_curve(opt_clf, X_train, y_train, 'hidden_layer_sizes', param_range, cv=10, n_jobs=-1)
av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)

# VC Plot
plt.plot(hidden_layer_two, av_train_scores, label='train scores')
plt.plot(hidden_layer_two, av_validation_scores, label='validation scores')
plt.title("Validation Curve")
plt.xlabel("Hidden Layer Two")
plt.ylabel("Scores")
plt.ylim([0.90, 1.0])
plt.legend()
plt.show()

# Alpha VC
opt_clf = MLPClassifier(activation='logistic', hidden_layer_sizes=[3,2], solver='lbfgs', random_state=random_seed)

param_range = np.linspace(0.01, 0.21, 20)
train_scores, validation_scores = validation_curve(opt_clf, X_train, y_train, 'alpha', param_range, cv=10, n_jobs=-1)
av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)

# VC Plot
plt.plot(param_range, av_train_scores, label='train scores')
plt.plot(param_range, av_validation_scores, label='validation scores')
plt.title("Validation Curve")
plt.xlabel("Alpha")
plt.ylabel("Scores")
plt.ylim([0.90, 1.0])
plt.legend()
plt.show()
