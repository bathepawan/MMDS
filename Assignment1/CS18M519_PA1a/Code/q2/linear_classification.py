
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir

from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pandas import read_csv
import numpy

class LinearClassification(object):
    def __init__(self, training_data_file, test_data_file):
        self.ds1_training_data = read_csv(training_data_file, sep=',', header=None,
                                          encoding='ISO-8859-1', engine='python')

        self.ds1_test_data = read_csv(test_data_file, sep=',', header=None, encoding='ISO-8859-1', engine='python')

        self.linear_classifier = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
        self.linear_classifier.fit(self.ds1_training_data.loc[:, 0:19], self.ds1_training_data.loc[:, 20])
        self.y_predicted = self.linear_classifier.predict(self.ds1_test_data.loc[:, 0:19])
        self.accuracy_score = accuracy_score(self.ds1_test_data.loc[:, 20], self.y_predicted)
        self.classification_report = classification_report(self.ds1_test_data.loc[:, 20], self.y_predicted)

    def retrieve_metrics(self):
        print('Regression Coefficients of Learned Model:')
        print(self.linear_classifier.coef_)
        numpy.savetxt('coeffs.csv', self.linear_classifier.coef_, delimiter=",")
        print('Accuracy Score: {}'.format(round(self.accuracy_score,3)))
        print('Classification Report: \n')
        print(self.classification_report)


if __name__=='__main__':
    training_file = os.path.join(dataset_dir, 'DS1-train.csv')
    test_file = os.path.join(dataset_dir, 'DS1-test.csv')

    classifier = LinearClassification(training_file, test_file)
    classifier.retrieve_metrics()
