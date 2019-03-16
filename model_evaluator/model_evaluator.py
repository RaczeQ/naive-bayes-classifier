import logging
from exceptions import NotEvaluatedError

from confusion_matrix import ConfusionMatrix
from naive_bayes import NaiveBayes


class ModelEvaluator(object):
    model = None
    train_set = None
    test_set = None
    main_cm = None
    class_cms = {}
    predictions = []
    classes = []

    def __init__(self, train_set, test_set, classes_list):
        self.train_set = train_set
        self.test_set = test_set
        self.classes = classes_list
        self.model = NaiveBayes(self.classes)

    def evaluate(self):
        self.model.fit(self.train_set)
        self.predictions = self.model.predict(self.test_set)
        self._generate_cms()
        logging.info("[Model Evaluator] Evaluated model")

    def get_f_score(self):
        if self.main_cm is None:
            raise NotEvaluatedError(self.__class__.__name__)
        return self.main_cm.f_score()

    def _generate_cm_for_class(self, meta_class):
        result = ConfusionMatrix()
        for idx, valid_class in enumerate(self.test_set.get_labels()):
            predicted_class = self.predictions[idx]
            if meta_class == valid_class:
                if valid_class == predicted_class:
                    result.TP += 1
                else:
                    result.FP += 1
            else:
                if valid_class == predicted_class:
                    result.TN += 1
                else:
                    result.FN += 1
        logging.info("[Model Evaluator] Calculated confusion matrix for '{}' class".format(meta_class))
        logging.debug("[Model Evaluator] Confusion Matrix\n{}".format(result))
        self.class_cms[meta_class] = result
        return result

    def _generate_cms(self):
        confusion_matrices = [self._generate_cm_for_class(c) for c in self.classes]
        result = ConfusionMatrix()
        for cm in confusion_matrices:
            result.TP += cm.TP
            result.TN += cm.TN
            result.FP += cm.FP
            result.FN += cm.FN
        logging.warning("[Model Evaluator] Calculated confusion matrix all classes")
        logging.info("[Model Evaluator] Confusion Matrix\n{}".format(result))
        self.main_cm = result
        return result
