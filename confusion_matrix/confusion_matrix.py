class ConfusionMatrix(object):
    TP = TN = FP = FN = 0

    def __str__(self):
        return "p\\t\tP\tN\nP\t{}\t{}\nN\t{}\t{}".format(self.TP, self.FP, self.FN, self.TN)
    
    def recall(self):
        return self.TP / float(self.TP + self.FN)
    
    def precision(self):
        return self.TP / float(self.TP + self.FP)

    def f_score(self):
        return 1.0 / ((1.0 / self.recall() + 1.0 / self.precision()) / 2.0)