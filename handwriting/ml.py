"""

Machine learning utility module.

"""

# Copyright (c) 2017 Ben Zimmer. All rights reserved.

from functools import partial
import inspect
import random

import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVC
from sklearn import neural_network

from handwriting import func
from handwriting.func import grid_search


#### Functional wrappers for sklearn models

# These functions take hyperparameters as inputs and return a fit function,
# which when called with features and or labels returns a predict function.

# Depending on the model type, the predict function may actually be a callable
# object that has the underlying sklearn model as an instance variable.


class CallableModel(object):
    """Wrapper for sklearn models to make them callable."""

    def __init__(self, model):
        """init method"""
        self.model = model

    def __call__(self, feats):
        """model becomes callable"""
        return self.model.predict(feats)


def feat_selection_pca(n_components):
    """PCA feature selection"""

    pca = PCA(n_components)

    def fit(feats):
        """fit model"""
        pca.fit(feats)
        print("PCA components:", pca.n_components_, "/", pca.n_features_)
        print("PCA variance explained:", np.sum(pca.explained_variance_ratio_))

        def select(feats_test):
            """perform feature selection"""
            return pca.transform(feats_test)

        return select
    return fit


def feat_scaler(robust):
    """Feature scaling"""

    if robust:
        scaler = sklearn.preprocessing.RobustScaler()
    else:
        scaler = sklearn.preprocessing.StandardScaler()

    def fit(feats):
        """fit model"""
        scaler.fit(feats)

        def scale(feats_test):
            """peform scaling"""
            return scaler.transform(feats_test)

        return scale
    return fit


def svc(c, gamma):
    """Build a function that fits an RBF SVC."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""

        model = SVC(C=c, gamma=gamma, probability=True)
        model.fit(feats_train, labels_train)

        return CallableModel(model)

    return fit


def linear_svc(c):
    """Build a function that fits a linear SVC."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""
        model = sklearn.svm.LinearSVC(C=c)
        model.fit(feats_train, labels_train)

        return CallableModel(model)

    return fit


def nn_classifier(hidden_layer_sizes, alpha):
    """Build a function that fits a neural network classifier."""

    def fit(feats_train, labels_train):
        """Perform the fitting."""

        # default model uses relu activation function, adam optimizer.
        # only loss function available is cross-entropy

        # TODO: experiment with learning rate

        model = neural_network.MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=alpha
            # tol=-1.0e16,
            # warm_start=True,
            # max_iter=1
        )

        # for i in range(10):
        #     for j in range(10):
        #         model.fit(feats_train, labels_train)
        #     print("  ", model.n_iter_, np.round(model.loss_, 5), flush=True)
        # print("->", end=" ", flush=True)

        model.fit(feats_train, labels_train)

        return CallableModel(model)

    return fit


def score_accuracy(model, feats_test, labels_test):

    """Calculate accuracy of an sklearn model."""

    labels_test_pred = model.predict(feats_test)
    score = sklearn.metrics.accuracy_score(labels_test, labels_test_pred)
    return score



def score_auc(model, feats_test, labels_test, decision_function=True):

    """Calculate ROC AUC of a sklearn classifier."""

    if decision_function:
        distances_test = model.decision_function(feats_test)
    else:
        # only correct for binary classifier
        predictions = model.predict_proba(feats_test)
        distances_test = predictions[:, 1]

    fpr, tpr, _ = sklearn.metrics.roc_curve(
        labels_test, distances_test)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    return roc_auc


# cross validation is a function which given a fit function, score function,
# features, and label (and optional callback), finds an aggregate score from
# fitting and scoring on different splits of the data

# TODO: functionality to abort a call based on bad properties of fit model

def kfold_cross_validation(n_splits):
    """Build a function to fit models using k-fold cross validation."""

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    return func.pipe(
        generic_validation(kfold),
        lambda x: np.mean(x) if len(x) == n_splits else None)


def holdout_validation(test_frac):
    """Build a function to fit models using holdout cross validation."""

    shufflesplit = ShuffleSplit(1, test_frac)

    return func.pipe(
        generic_validation(shufflesplit),
        lambda x: x[0])


def generic_validation(cross_validator):
    """Fit models using sklearn cross validators."""

    def cross_validate(
            fit_func, score_func, feats, labels,
            fold_callback=None):

        """Perform the cross validation."""

        scores = []
        for idxs_train, idxs_test in cross_validator.split(feats):
            feats_train = [feats[idx] for idx in idxs_train]
            labels_train = [labels[idx] for idx in idxs_train]
            feats_test = [feats[idx] for idx in idxs_test]
            labels_test = [labels[idx] for idx in idxs_test]

            model = fit_func(
                feats_train, labels_train)

            score = score_func(model.model, feats_test, labels_test)

            if fold_callback is not None:
                fold_callback(score)

            if score is None:
                break
            scores.append(score)

        return scores

    return cross_validate


def train_classifier(
        build_fit_model,
        cross_validation,
        score_func,
        feats, labels,
        **grid_search_params):

    """train a classifier"""

    def params_to_string(params):
        """convert list of params to string"""
        def fix(val):
            """helper"""
            if isinstance(val, float):
                val = np.round(val, 5)
            return val
        return " ".join([str(k) + "=" + str(fix(v)) for k, v in params])

    def grid_search_callback(param_set, result):
        """log the result for the current hyperparamers to the screen"""
        print("final:", params_to_string(param_set.items()) + ":", result)
        print("---")

    cross_validate = partial(
        cross_validation,
        score_func=score_func,
        feats=feats,
        labels=labels)

    def fit_model_with_cross_validation(*args, **kwargs):
        """helper to build parameter string for fold callback log"""
        # get ordered list of fit function parameters
        param_values_list = []
        for param_name in func.function_param_order(build_fit_model):
            if param_name in kwargs.keys():
                param_values_list.append((param_name, kwargs[param_name]))
        def fold_callback(score):
            print(params_to_string(param_values_list) + ":", score)
        fit_model = build_fit_model(*args, **kwargs)
        return cross_validate(fit_model, fold_callback=fold_callback)

    # this is awkward
    fit_model_with_cross_validation.__signature__ = inspect.signature(build_fit_model)

    models = grid_search(
        fit_model_with_cross_validation,
        gs_callback=grid_search_callback,
        **grid_search_params)

    models = [x for x in models if x[1] is not None]

    best_params, mean_score = models[np.argmax([x[1] for x in models])]

    print("best mean score:", mean_score)
    print("best hyperparameters:", best_params)
    model = build_fit_model(**best_params)(feats, labels)

    return model


def classification_process(feat_extractor, feat_selector, classifier):
    """combine feature extraction, feature selection, and classification"""

    def predict(data):
        """helper"""
        feats = feat_selector([feat_extractor(x) for x in data])
        return classifier(feats)

    return predict


def group_by_label(samples, labels):
    """group samples by label"""

    unique_labels = sorted(list(set(labels)))

    grouped = [[x for x, y in zip(samples, labels) if y == label]
               for label in unique_labels]

    res_unsorted = list(zip(unique_labels, grouped))
    # order = np.argsort([len(x) for x in grouped])
    order = sorted(range(len(grouped)), key=lambda k: unique_labels[k])
    return [res_unsorted[idx] for idx in order]


def balance(samples, labels, balance_factor, adjust_func):
    """create a balanced dataset by subsampling classes or generating new
    samples"""

    grouped = group_by_label(samples, labels)

    if balance_factor <= 1.0:
        largest_group_size = max([len(x[1]) for x in grouped])
        target_group_size = int(largest_group_size * balance_factor)
    else:
        target_group_size = int(balance_factor)

    grouped_balanced = []
    for label, group in grouped:

        if len(group) > target_group_size:
            print(label, 1.0)
            group_resized = random.sample(group, target_group_size)
        else:
            print(label, (len(group) * 1.0) / target_group_size)
            group_resized = [x for x in group]
            while len(group_resized) < target_group_size:
                group_resized.append(adjust_func(random.choice(group)))
        grouped_balanced.append((label, group_resized))

    pairs = [(y, x[0]) for x in grouped_balanced for y in x[1]]
    return zip(*pairs)
