from data_model import DataModel
from data_utils import IRIS_DATASET, PIMA_DIABETES_DATASET, GLASS_DATASET, WINE_DATASET, SUGGESTED_DISCRETIZE_FEATURES_KEY
from math_utils import bucket_discretize
from model_evaluator import ModelEvaluator

import logging

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.ERROR)

def main():
    dm = DataModel.generate_from_file(PIMA_DIABETES_DATASET, 
            smooth=True,
            discretize=False,
            discretize_function=bucket_discretize,
            discrete_columns=[]
        )
    classes_list = dm.get_classes_list()
    models = []
    for _ in range(5):
        for train_set, test_set in dm.generate_k_folds_strategized(2):
            model_evaluator = ModelEvaluator(train_set, test_set, classes_list)
            model_evaluator.evaluate()
            print("F score: {}".format(model_evaluator.get_f_score()))
            models.append(model_evaluator)
    
    models.sort(key=lambda x: x.get_f_score(), reverse=True)
    best_model = models[0]
    print("Best model: {}".format(best_model.get_f_score()))
    print(best_model.main_cm)

if __name__ == '__main__':
    main()