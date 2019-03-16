from data_model import DataModel
from data_utils import IRIS_DATASET, PIMA_DIABETES_DATASET, GLASS_DATASET, WINE_DATASET
from data_visualizer import visualize
from math_utils import DiscretizeParam, bucket_discretize, frequency_discretize, kbins_discretize
from model_evaluator import ModelEvaluator
from parameters_tester import Tester
from datetime import datetime

import logging

LOG_FILENAME = 'logs/{}.log'.format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
logging.basicConfig(filename=LOG_FILENAME, format='%(asctime)s - %(message)s', level=logging.ERROR)
logging.getLogger().addHandler(logging.StreamHandler())

def main():
    dm = DataModel.generate_from_file(PIMA_DIABETES_DATASET, 
            smooth=True,
            discretize_params=[
                DiscretizeParam('Age', bucket_discretize, 6)
            ]
        )
    classes_list = dm.get_classes_list()
    models = []
    for _ in range(5):
        for train_set, test_set in dm.generate_k_folds(2):
            model_evaluator = ModelEvaluator(train_set, test_set, classes_list)
            model_evaluator.evaluate()
            print("F score: {}".format(model_evaluator.get_f_score()))
            models.append(model_evaluator)
    
    models.sort(key=lambda x: x.get_f_score(), reverse=True)
    best_model = models[0]
    print("Best model: {}".format(best_model.get_f_score()))
    print(best_model.main_cm)

if __name__ == '__main__':
    try:
        # main()
        # visualize(WINE_DATASET)
        Tester()
    except Exception as e:
        logging.exception('Got exception on main handler')
        logging.error(e, exc_info=True)
        raise