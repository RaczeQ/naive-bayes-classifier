import logging
from collections import Counter
from exceptions import EmptyDiscretizeFunctionError

import numpy as np
import pandas as pd

from math_utils import DiscretizeParam, bucket_discretize


class Dataset(object):
    name = None
    path = None
    continuous_features = []
    discrete_features = []

    def __init__(self, name, path, continuous_features = [], discrete_features = [], suggested_discretize_features = []):
        self.name = name
        self.path = path
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        assert all([suggested in continuous_features for suggested in suggested_discretize_features])
        self.suggested_discretize_features = suggested_discretize_features

    def __repr__(self):
        return self.name

IRIS_DATASET = Dataset(
    name = 'iris',
    path = 'datasets/iris.csv',
    continuous_features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
    suggested_discretize_features = ['PetalLength', 'PetalWidth']
)

PIMA_DIABETES_DATASET = Dataset(
    name = 'diabetes',
    path = 'datasets/diabetes.csv',
    continuous_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'BMI', 'DiabetesPedigreeFunction'],
    suggested_discretize_features = ['Pregnancies', 'Age', 'SkinThickness']
)

GLASS_DATASET = Dataset(
    name = 'glass',
    path = 'datasets/glass.csv',
    continuous_features = ['RefractiveIndex', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron'],
    suggested_discretize_features = ['RefractiveIndex', 'Magnesium', 'Potassium', 'Barium', 'Iron']
)

WINE_DATASET = Dataset(
    name = 'wine',
    path = 'datasets/wine.csv',
    continuous_features = ['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280/OD315', 'Proline'],
    suggested_discretize_features = ['Alcohol', 'MalicAcid', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'ColorIntensity', 'Hue', 'OD280/OD315', 'Proline']
)

def load_csv(file_path):
    df = pd.read_csv(file_path)
    logging.warning("[Data Processor] Loaded file '{}' with {} rows".format(
        file_path, len(df.index)))
    return df


def discretize_continuous_features(df, discretize_params):
    for dp in discretize_params:
        col = dp.feature_name
        function = dp.discretize_function
        buckets = dp.buckets_amount
        df[col] = df[col].apply(lambda x: function(list(df[col]), x, buckets))
        logging.warning("[Data Processor] Discretized '{}' column using '{}' function into {} values.".format(
            col, function.__name__, buckets))
    return df


def split_data_by_class(df):
    dfs = dict(tuple(df.groupby(df.iloc[:, -1])))
    logging.info(
        "[Data Processor] Splitted data into {} classes".format(len(dfs.keys())))
    return dfs
