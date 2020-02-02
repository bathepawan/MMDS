import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv


class KNNClassification(object):
    def __init__(self, training_data_file, test_data_file, k=2):
        self.ds1_training_data = read_csv(training_data_file, sep=',', header=None,
                                          encoding='ISO-8859-1', engine='python')
        self.ds1_test_data = read_csv(test_data_file, sep=',', header=None, encoding='ISO-8859-1', engine='python')
        self.knn_classifier = KNeighborsClassifier(n_neighbors=k, p=5, weights='distance')
        self.knn_classifier.fit(self.ds1_training_data.loc[:, 0:19], self.ds1_training_data.loc[:, 20])
        self.y_predicted = self.knn_classifier.predict(self.ds1_test_data.loc[:, 0:19])
        self.accuracy_score = accuracy_score(self.ds1_test_data.loc[:, 20], self.y_predicted)
        self.classification_report = classification_report(self.ds1_test_data.loc[:, 20], self.y_predicted)

    def retrieve_metrics(self):
        print('Accuracy Score: {}'.format(self.accuracy_score))
        print('Classification Report: \n')
        print(self.classification_report)


if __name__=='__main__':
    training_file = os.path.join(dataset_dir, 'DS1-train.csv')
    test_file = os.path.join(dataset_dir, 'DS1-test.csv')
    for k in range(30, 35):
        print('Report For K={}:'.format(k))
        classifier = KNNClassification(training_file, test_file, k=k)
        classifier.retrieve_metrics()

