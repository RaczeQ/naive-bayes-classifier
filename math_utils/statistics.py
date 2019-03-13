import math 
import numpy as np
import logging

def calculate_probability(x, mean, std):
    logging.debug("[Naive Bayes Model] Calculating probability: ({},{},{})".format(x, mean, std))
    result = 0.0
    if std > 0.0:
        exponent = np.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
        result = (1 / (math.sqrt(2*math.pi) * std)) * exponent
        logging.info("[Naive Bayes Model] Calculated probability: ({},{},{}) = {}".format(x, mean, std, result))
    return result
    # [Naive Bayes Model] Calculated probability: (1.52101,11.518105999999998,0.0019803852439073517) = 0.0
    # [Naive Bayes Model] Calculated probability: (71.78,73.31125,1.104179559673153) = 0.13812236139021056