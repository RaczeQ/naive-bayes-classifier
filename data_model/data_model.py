import logging
from collections import Counter
from exceptions import EmptyDiscretizeFunctionError

import numpy as np
import pandas as pd

from data_utils import (discretize_continuous_features, load_csv,
                        split_data_by_class)


class DataModel(object):
    _df = None
    _discrete_features = []
    _discrete_features_values = {}
    _continuous_features = []
    smooth = True

    def __init__(self, data_frame, discrete_features, discrete_features_values, continous_features, smooth):
        self._df = data_frame
        self.smooth = smooth
        self._discrete_features = discrete_features
        self._discrete_features_values = discrete_features_values
        self._continuous_features = continous_features

    @classmethod
    def generate_from_file(self, dataset, smooth=True, discretize_params=[]):
        _df = load_csv(dataset.path)
        _smooth = smooth
        _continuous_features = dataset.continuous_features
        _discrete_features = dataset.discrete_features
        if len(discretize_params) > 0:
            discrete_columns = [dp.feature_name for dp in discretize_params]
            _discrete_features = list(
                set(_discrete_features + discrete_columns))
            _continuous_features = [
                c for c in _continuous_features if not c in discrete_columns]
            _df = discretize_continuous_features(
                _df, discretize_params)
            
        _discrete_features_values = {}
        for discrete_feature in _discrete_features:
            _discrete_features_values[discrete_feature] = list(pd.unique(_df[discrete_feature].values.ravel('K')))
        return self(_df, _discrete_features, _discrete_features_values, _continuous_features, smooth)

    def get_design_matrix(self):
        return self._df.iloc[:, :-1]

    def get_classes_column(self):
        return self._df.iloc[:, -1]

    def get_classes_list(self):
        return pd.unique(self.get_classes_column().values.ravel('K'))

    def get_classes_counts(self):
        return Counter(self.get_classes_column())

    def get_priors(self, classes_list=[]):
        if len(classes_list) == 0:
            classes_list = self.get_classes_list()
        classes_column = self.get_classes_column()
        freq = self.get_classes_counts()
        return { c:(freq[c] if c in freq else 0)/len(classes_column) for c in classes_column }

    def get_features(self, classes_list=[]):
        # discrete: (class : feature : value : likelihood)
        # continuous: (class : feature : (mean, std))
        if len(classes_list) == 0:
            classes_list = self.get_classes_list()
        cont_feat_dict = {c:{} for c in classes_list}
        disc_feat_dict = {c:{} for c in classes_list}
        dfs = dict(tuple(self._df.groupby(self.get_classes_column())))
        for c in classes_list:
            cont_feat_dict[c] = {}
            disc_feat_dict[c] = {}
            total_rows = len(dfs[c].index)
            for f in self._discrete_features_values.keys():
                disc_feat_dict[c][f] = {}
                f_col = dfs[c][f]
                f_counts = Counter(f_col)
                f_unique_values = f_counts.keys()
                for v in self._discrete_features_values[f]:
                    if self.smooth:
                        disc_feat_dict[c][f][v] = float(f_counts[v] + 1) / float(total_rows + len(f_unique_values))
                    else:
                        disc_feat_dict[c][f][v] = float(f_counts[v]) / float(total_rows)
            for f in self._continuous_features:
                cont_feat_dict[c][f] = {}
                f_col = dfs[c][f]
                desc = f_col.describe()
                cont_feat_dict[c][f]['mean'] = desc['mean']
                cont_feat_dict[c][f]['std'] = desc['std']
        return disc_feat_dict, cont_feat_dict

    def generate_k_folds(self, k_fold):
        folds = np.array_split(self._df.sample(frac=1), k_fold)
        for k in range(k_fold):
            train = folds.copy()
            test = folds[k]
            del train[k]
            train = pd.concat(train, sort=False)
            train_obj = DataModel(train, self._discrete_features, self._discrete_features_values, self._continuous_features, self.smooth)
            test_obj = DataModel(test, self._discrete_features, self._discrete_features_values, self._continuous_features, self.smooth)
            yield train_obj, test_obj

    def generate_k_folds_stratified(self, k_fold):
        folds_dict = {k: np.array_split(
            data, k_fold) for k, data in split_data_by_class(self._df.sample(frac=1)).items()}
        folds = [pd.concat([v[k] for _, v in folds_dict.items()],
                           sort=False) for k in range(k_fold)]
        for k in range(k_fold):
            train = folds.copy()
            test = folds[k]
            del train[k]
            train = pd.concat(train, sort=False)
            train_obj = DataModel(train, self._discrete_features, self._discrete_features_values, self._continuous_features, self.smooth)
            test_obj = DataModel(test, self._discrete_features, self._discrete_features_values, self._continuous_features, self.smooth)
            yield train_obj, test_obj

    def generate_vector_and_label(self):
        for i in range(len(self._df.index)):
            yield self.get_design_matrix().iloc[i], self.get_classes_column().iloc[i]

    def get_labels(self):
        return list(self.get_classes_column())

    def get_splitted_dataframe(self):
        return split_data_by_class(self._df)
            
    def __str__(self):
        return str(self._df)
