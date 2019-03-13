import logging
from collections import Counter
from exceptions import EmptyDiscretizeFunctionError

import numpy as np
import pandas as pd

PATH_KEY = 'path'
CONTINUOUS_FEATURES_KEY = 'continuous'
DISCRETE_FEATURES_KEY = 'discrete'
SUGGESTED_DISCRETIZE_FEATURES_KEY = 'suggested_discretize'

IRIS_DATASET = {
    PATH_KEY: 'datasets/iris.csv',
    CONTINUOUS_FEATURES_KEY: ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'],
    DISCRETE_FEATURES_KEY: [],
    SUGGESTED_DISCRETIZE_FEATURES_KEY: []
}
PIMA_DIABETES_DATASET = {
    PATH_KEY: 'datasets/diabetes.csv',
    CONTINUOUS_FEATURES_KEY: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'Age', 'BMI', 'DiabetesPedigreeFunction'],
    DISCRETE_FEATURES_KEY: [],
    SUGGESTED_DISCRETIZE_FEATURES_KEY: []
}
GLASS_DATASET = {
    PATH_KEY: 'datasets/glass.csv',
    CONTINUOUS_FEATURES_KEY: ['RefractiveIndex', 'Sodium', 'Magnesium', 'Aluminum', 'Silicon', 'Potassium', 'Calcium', 'Barium', 'Iron'],
    DISCRETE_FEATURES_KEY: [],
    SUGGESTED_DISCRETIZE_FEATURES_KEY: ['Potassium']
}
WINE_DATASET = {
    PATH_KEY: 'datasets/wine.csv',
    CONTINUOUS_FEATURES_KEY: ['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280/OD315', 'Proline'],
    DISCRETE_FEATURES_KEY: []
}


def load_csv(file_path):
    df = pd.read_csv(file_path)
    logging.warning("[Data Processor] Loaded file '{}' with {} rows".format(
        file_path, len(df.index)))
    return df


def discretize_continuous_features(df, function, columns):
    for col in columns:
        min_value = df[col].min()
        max_value = df[col].max()
        df[col] = df[col].apply(lambda x: function(min_value, max_value, x))
    logging.warning("[Data Processor] Discretized {} columns using {} function.".format(
        len(columns), function.__name__))
    return df


def split_data_by_class(df):
    dfs = dict(tuple(df.groupby(df.iloc[:, -1])))
    logging.info(
        "[Data Processor] Splitted data into {} classes".format(len(dfs.keys())))
    return dfs
