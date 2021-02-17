import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import tree, svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
plt.rcParams["figure.figsize"] = (3.5,2.5)

random_seed = 17

ds = ds.make_classification(n_samples=10000, n_features=10, n_informative=5, n_repeated=2,
                            n_clusters_per_class=5, flip_y=0.025, class_sep = 0.5, random_state=random_seed)

X, y = ds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)


# First gridsearch
param_dict = {
    'base_estimator__criterion' : ['gini', 'entropy'],
    'base_estimator__max_depth' : np.arange(2,11),
    'n_estimators': [10,50,100,500,1000],
  }

clf = AdaBoostClassifier(tree.DecisionTreeClassifier(random_state=random_seed))
grid_object = GridSearchCV(estimator = clf, param_grid = param_dict, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_object.fit(X_train, y_train)
best_params = grid_object.best_params_
print(best_params)

opt_clf_inner = tree.DecisionTreeClassifier(criterion='entropy', max_depth=9, random_state= random_seed)
opt_clf = AdaBoostClassifier(opt_clf_inner, n_estimators=500, random_state=random_seed)

# Learning Curve Plots
train_sizes, train_scores, validation_scores = learning_curve(opt_clf, X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 100), n_jobs=-1)
av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)
# LC Plot
plt.plot(train_sizes, av_train_scores, label='train scores')
plt.plot(train_sizes, av_validation_scores, label ='validation scores')
plt.title("Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Scores")
plt.ylim([0.90, 1.02])
plt.legend()
plt.show()

# Validation Curves
# Max Depth
# param_range = np.arange(1,21)
# train_scores, validation_scores = validation_curve(opt_clf, X_train, y_train, 'base_estimator__max_depth', param_range, cv=10, n_jobs=-1)

# # N Estimators
opt_clf_inner = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state= random_seed)
opt_clf = AdaBoostClassifier(opt_clf_inner, random_state=random_seed)

param_range = np.arange(1,201)
train_scores, validation_scores = validation_curve(opt_clf, X_train, y_train, 'n_estimators', param_range, cv=10, n_jobs=-1)

av_train_scores = np.mean(train_scores, axis=1)
av_validation_scores = np.mean(validation_scores,axis=1)

# VC Plot
plt.plot(param_range, av_train_scores, label='train scores')
plt.plot(param_range, av_validation_scores, label='validation scores')
plt.title("Validation Curve")
plt.ylabel("Scores")
plt.ylim([0.90, 1.02])
plt.legend()
plt.show()