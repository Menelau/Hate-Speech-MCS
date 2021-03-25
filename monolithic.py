from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score
#
from sklearn.model_selection import ParameterGrid

from sklearn.pipeline import Pipeline

import copy

class Mestrado:

    classifiers_ = None
    params_ = None
    train = None
    class_train = None
    val = None
    class_val = None

    def __init__(self, train, class_train, val, class_val):
        self.classifiers_ = self.get_classifiers()
        self.params_ = self.get_parameters()
        self.train = train
        self.class_train = class_train
        self.val = val
        self.class_val = class_val

    def get_classifiers(self):
        classifiers = {
            'SVM': SVC(random_state=42, verbose=100, probability=True),
            'LR': LogisticRegression(random_state=42, verbose=100, multi_class='auto', solver='liblinear'),
            'RF': RandomForestClassifier(random_state=42, verbose=100),
            'MNB': MultinomialNB(),
            'BNB': BernoulliNB(),
            'MLP': MLPClassifier(random_state=42, batch_size=20, max_iter=20, verbose=100),
            'EXTRA': ExtraTreesClassifier(random_state=42, verbose=100),
            'KNN': KNeighborsClassifier(n_neighbors=3)
        }
        return classifiers

    def get_parameters(clf__kernelself):
        params = {
            'SVM':{
                'kernel': ['linear', 'sigmoid', 'rbf'],
                'gamma': [0.1, 1, 0.5]
            },
            'LR': {
                'penalty': ['l1', 'l2']
            },
            'RF': {
                'n_estimators': [10, 20, 50]
            },
            'MNB': {
                'alpha': [0.1, 0.5, 1],
                'fit_prior': [False, True]
            },
            'BNB': {
                'alpha': [0.1, 0.5, 1],
                'fit_prior': [False, True]
            },
            'MLP': {
                'activation': ['relu', 'logistic'],
                'solver': ['adam', 'lbfgs']
            },
            'EXTRA': {
                'n_estimators': [10, 20, 50]
            },
            'KNN': {
                'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'n_neighbors': [3, 5]
            }
        }
        return params

    def get_param_classifier(self, classifier):
        return list(ParameterGrid(self.params_[classifier]))

    def fit_params(self, classifier):
        clf = self.classifiers_[classifier].set_params(
            self.params_[classifier]
        )
        return clf.fit(self.train, self.class_train)


    def fit_all(self, classifier=None):
        estimators = []
        params_clf = self.get_param_classifier(classifier)
        total_params = len(params_clf)
        k = 1

        for params in params_clf:
            classifiers = copy.deepcopy(self.classifiers_)
            clf = classifiers[classifier]
            clf.set_params(**params)
            estimators.append(clf.fit(self.train, self.class_train))
            print("Feito {} de {}".format(k, total_params))
            k += 1
        return estimators

    def best_estimator(self, estimators):
        best_score = 0
        best_estimator = None
        for e in estimators:
            # y_pred = e.predict(self.val)
            # acc = accuracy_score(self.class_val, y_pred)
            score = e.score(self.val, self.class_val)
            if best_score < score:
                best_estimator = e
                best_score = score
        return best_estimator
