import itertools
import logging
from datetime import datetime

import pandas as pd

from data_model import DataModel
from data_utils import (GLASS_DATASET, IRIS_DATASET, PIMA_DIABETES_DATASET,
                        WINE_DATASET)
from math_utils import (DiscretizeParam, bucket_discretize,
                        frequency_discretize, kbins_discretize)
from model_evaluator import ModelEvaluator

DATASETS = [IRIS_DATASET, PIMA_DIABETES_DATASET, GLASS_DATASET, WINE_DATASET]
FOLDS = [(2, 75), (3, 50), (5, 30), (10, 15)] # Always test 150 models
BUCKETS = range(3, 11)
DISCRETIZE_FUNCTIONS = [bucket_discretize, frequency_discretize, kbins_discretize]


class Tester(object):
    result_df = pd.DataFrame(columns=['dataset', 'fold', 'f_score', 'permutation', 'feature', 'function', 'bins'])
    file_no = 0
    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    best_fcs = { d:0 for d in DATASETS }
    best_fold = { d:None for d in DATASETS }
    best_discretize_feature_params = { d:[] for d in DATASETS }
    best_discretize_parameters = { d:[] for d in DATASETS }

    def __init__(self):
        self.test()

    def clear_result(self):
        self.result_df = self.result_df.iloc[0:0]

    def save_result(self): 
        file_path = 'test_results/{}_{}.csv'.format(self.file_name, self.file_no)
        self.result_df.to_csv(file_path, encoding='utf-8')
        self.file_no += 1
        logging.error("[Parameters Tester] Saved result to path {} with {} rows".format(file_path, len(self.result_df.index))) 
        self.clear_result()

    def append_result(self, row):
        self.result_df = self.result_df.append(row, ignore_index=True)
        if len(self.result_df.index) == 5000:
            self.save_result()

    def test(self):
        try:
            for dataset in DATASETS:
                self.find_best_fold(dataset)
                self.find_best_single_feature_parameters(dataset)
                self.find_best_parameters(dataset)
                self.save_result()
            logging.error(self.best_fold)
            logging.error(self.best_discretize_parameters)
        except:
            self.save_result()
            raise

    def find_best_fold(self, dataset):
        dm = DataModel.generate_from_file(dataset)
        classes_list = dm.get_classes_list()
        for fold in FOLDS:
            f_scores = []
            a = 1
            for _ in range(fold[1]):
                for train_set, test_set in dm.generate_k_folds_stratified(fold[0]):
                    model_evaluator = ModelEvaluator(train_set, test_set, classes_list)
                    model_evaluator.evaluate()
                    f_scores.append(model_evaluator.get_f_score())
                    logging.error("[Parameters Tester][{}][CV{:02d}][{:03d}] FCS: {}".format(dataset, fold[0], a, f_scores[-1]))
                    a += 1
            f_score_mean = sum(f_scores) / len(f_scores)
            logging.error("[Parameters Tester][{}][CV{:02d}] Best FCS: {}, Mean FCS {}".format(dataset, fold[0], max(f_scores), f_score_mean))
            self.append_result({'dataset':dataset.name, 'fold':fold[0], 'f_score':f_score_mean, 'permutation':-1})
            if f_score_mean > self.best_fcs[dataset]:
                self.best_fold[dataset] = fold
                self.best_fcs[dataset] = f_score_mean
        logging.error("[Parameters Tester][{}] Best mean FCS: {}, Best fold: {}".format(dataset, self.best_fcs[dataset], self.best_fold[dataset]))   

    def generate_feature_parameters(self, feature):
        parameters = []
        for function in DISCRETIZE_FUNCTIONS:
            for buckets_amount in BUCKETS:
                parameters.append([DiscretizeParam(
                    feature_name=feature,
                    discretize_function=function,
                    buckets_amount=buckets_amount
                )])
        return parameters

    def find_best_single_feature_parameters(self, dataset):
        for feature in dataset.suggested_discretize_features:
            permutations = self.generate_feature_parameters(feature)
            print(permutations)
            best_mean_fcs = self.best_fcs[dataset]
            best_perm = None
            for p, perm in enumerate(permutations):
                logging.error("[Parameters Tester][{}][{}][Perm {:03d}] Current permutation: {}".format(dataset, feature, p+1,  perm))
                dm = DataModel.generate_from_file(dataset, discretize_params=perm)
                classes_list = dm.get_classes_list()
                f_scores = []
                a = 1
                for _ in range(self.best_fold[dataset][1]):
                    for train_set, test_set in dm.generate_k_folds_stratified(self.best_fold[dataset][0]):
                        model_evaluator = ModelEvaluator(train_set, test_set, classes_list)
                        model_evaluator.evaluate()
                        f_scores.append(model_evaluator.get_f_score())
                        logging.error("[Parameters Tester][{}][{}][Perm {:03d}][{:03d}] FCS: {}".format(dataset, feature, p+1, a, f_scores[-1]))
                        a += 1
                f_score_mean = sum(f_scores) / len(f_scores)
                logging.error("[Parameters Tester][{}][{}][Perm {:03d}] Best FCS: {}, Mean FCS {}".format(dataset, feature, p+1, max(f_scores), f_score_mean))
                if f_score_mean > best_mean_fcs:
                    best_perm = perm[0]
                    best_mean_fcs = f_score_mean
            if best_perm is not None:
                self.best_discretize_feature_params[dataset].append(best_perm)
            logging.error("[Parameters Tester][{}][{}] Best mean FCS: {}, Best parameters: {}".format(dataset, feature, best_mean_fcs, best_perm))

    def generate_permutations(self, dataset):
        permutations = []
        for i in range(1, len(self.best_discretize_feature_params[dataset]) + 1):
            permutations += itertools.combinations(self.best_discretize_feature_params[dataset], i)
        print(self.best_discretize_feature_params[dataset])
        print(permutations)
        logging.error("[Parameters Tester][{}] Generated {} permutations".format(dataset, len(permutations)))
        return permutations

    def find_best_parameters(self, dataset):
        permutations = self.generate_permutations(dataset)
        for p, perm in enumerate(permutations):
            logging.error("[Parameters Tester][{}][Perm {:08d}] Current permutation: {}".format(dataset, p+1, perm))
            dm = DataModel.generate_from_file(dataset, discretize_params=perm)
            classes_list = dm.get_classes_list()
            f_scores = []
            a = 1
            for _ in range(self.best_fold[dataset][1]):
                for train_set, test_set in dm.generate_k_folds_stratified(self.best_fold[dataset][0]):
                    model_evaluator = ModelEvaluator(train_set, test_set, classes_list)
                    model_evaluator.evaluate()
                    f_scores.append(model_evaluator.get_f_score())
                    logging.error("[Parameters Tester][{}][Perm {:08d}][{:03d}] FCS: {}".format(dataset, p+1, a, f_scores[-1]))
                    a += 1
            f_score_mean = sum(f_scores) / len(f_scores)
            logging.error("[Parameters Tester][{}][Perm {:08d}] Best FCS: {}, Mean FCS {}".format(dataset, p+1, max(f_scores), f_score_mean))
            for param in perm:
                self.append_result({'dataset':dataset.name, 'fold':self.best_fold[dataset][0], 'f_score':f_score_mean, 'permutation':p + 1, 'feature':param.feature_name, 'function':param.discretize_function.__name__, 'bins':param.buckets_amount})
            if f_score_mean > self.best_fcs[dataset]:
                self.best_discretize_parameters[dataset] = perm
                self.best_fcs[dataset] = f_score_mean
        logging.error("[Parameters Tester][{}] Best mean FCS: {}, Best parameters: {}".format(dataset, self.best_fcs[dataset], self.best_discretize_parameters[dataset]))
