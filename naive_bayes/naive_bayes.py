import logging
from exceptions import NotFittedError
from math import log

import numpy as np

from confusion_matrix import ConfusionMatrix
from math_utils import calculate_probability


def is_fitted(func):
    def wrapper(self, *arg):
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return func(self, *arg)
    return wrapper

class NaiveBayes(object):
    classes_list = []
    priors = {}
    discrete_features = {}
    continuous_features = {}
    _is_fitted = False

    def __init__(self, classes_list):
        self.classes_list = classes_list

    def fit(self, data_model):
        self.priors = data_model.get_priors(self.classes_list)
        self.discrete_features, self.continuous_features = data_model.get_features(self.classes_list)
        logging.info("[Naive Bayes Model] Fitted model")
        self._is_fitted = True

    @is_fitted
    def predict_record(self, vector):
        prob = {k: log(v) for k, v in self.priors.items()}
        for c in self.classes_list:
            for f in self.discrete_features[c].keys():
                value = self.discrete_features[c][f][vector[f]]
                logging.info('Discrete [{}][{}][{}] = {}'.format(c, f, vector[f], value))
                if value > 0:
                    prob[c] += log(value)
            for f in self.continuous_features[c].keys():
                value = calculate_probability(vector[f], self.continuous_features[c][f]['mean'], self.continuous_features[c][f]['std'])
                logging.info('Continuous [{}][{}][{}] = {}'.format(c, f, vector[f], value))
                if value > 0:
                    prob[c] += log(value)
        return max(prob, key=prob.get)

    @is_fitted
    def predict(self, data_model):
        predictions = []
        for vector, _ in data_model.generate_vector_and_label():
            result = self.predict_record(vector)
            predictions.append(result)
        return predictions
