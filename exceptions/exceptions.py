class EmptyDiscretizeFunctionError(ValueError):
    """Raise in place of empty discretize function when loading dataset."""

    def __init__(self):
        message = self.message()
        super(EmptyDiscretizeFunctionError, self).__init__(message)

    @staticmethod
    def message():
        return "Please pass discretization method in DataModel contructor when using discretize = True."

class NotFittedError(ValueError):
    """Raise if predict is called before fit."""

    def __init__(self, class_name):
        message = self.message(class_name)
        super(NotFittedError, self).__init__(message)

    @staticmethod
    def message(class_name):
        return ("This instance of " + class_name +
                " has not been fitted yet. Please call "
                "'fit' before you call 'predict'.")
                
class NotEvaluatedError(ValueError):
    """Raise if get_f_score is called before evaluate."""

    def __init__(self, class_name):
        message = self.message(class_name)
        super(NotEvaluatedError, self).__init__(message)

    @staticmethod
    def message(class_name):
        return ("This instance of " + class_name +
                " has not been evaluate yet. Please call "
                "'evaluate' before you call 'get_f_score'.")
