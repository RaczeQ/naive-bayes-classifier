import logging
import math

import numpy as np


def calculate_probability(x, mean, std):
    logging.debug("[Naive Bayes Model] Calculating probability: ({},{},{})".format(x, mean, std))
    result = 0.0
    if std > 0.0:
        exponent = np.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        result = (1 / (math.sqrt(2*math.pi) * std)) * exponent
        logging.debug("[Naive Bayes Model] Calculated probability: ({},{},{}) = {}".format(x, mean, std, result))
    return result
