from IPython.display import Image
from subprocess import call

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_graphviz

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings


def c_plots(est, name, clfs, x_tr, y_tr, x_te, y_te):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    fig.suptitle(f"Classifier params: {name}")
    best_scores = {}

    for ax, (clf_name, clf) in zip(axs.flatten(), clfs.items()):
        n_range = range(5, 100)
        train_vals = []
        test_vals = []
        for i in n_range:
            model = est(clf(), n_estimators=i, random_state=42)
            model.fit(x_tr, y_tr)
            train_vals.append(
                roc_auc_score(y_tr, model.predict_proba(x_tr)[:, 1]))
            test_vals.append(
                roc_auc_score(y_te, model.predict_proba(x_te)[:, 1]))
        best_scores[clf_name] = max(test_vals)
        ax.plot(n_range, train_vals, label="train_score")
        ax.plot(n_range, test_vals, label="test_score")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Score")
        ax.set_title(clf_name)
        ax.legend()
    best_score = max(best_scores.values())
    rev_dict = {v: k for k, v in best_scores.items()}
    return rev_dict[best_score], best_score


def r_plots(est, name, clfs, x_tr, y_tr, x_te, y_te):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 5))
    fig.suptitle(f"Regression params: {name}")
    best_scores = {}

    for ax, (clf_name, clf) in zip(axs.flatten(), clfs.items()):
        n_range = range(5, 100)
        train_vals = []
        test_vals = []
        for i in n_range:
            model = est(clf(), n_estimators=i, random_state=42)
            model.fit(x_tr, y_tr)
            train_vals.append(
                model.score(x_tr, y_tr))
            test_vals.append(
                model.score(x_te, y_te))
        best_scores[clf_name] = max(test_vals)
        ax.plot(n_range, train_vals, label="train_score")
        ax.plot(n_range, test_vals, label="test_score")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("Score")
        ax.set_title(clf_name)
        ax.legend()
    best_score = max(best_scores.values())
    rev_dict = {v: k for k, v in best_scores.items()}
    return rev_dict[best_score], best_score



