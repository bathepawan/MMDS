import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.q3.knn_classification import KNNClassification
from Code.common.config import dataset_dir


if __name__=='__main__':
    training_file = os.path.join(dataset_dir, 'DS1-train.csv')
    test_file = os.path.join(dataset_dir, 'DS1-test.csv')
    for k in range(30, 35):
        print('Report For K={}:'.format(k))
        classifier = KNNClassification(training_file, test_file, k=k)
        classifier.retrieve_metrics()

    print('Report For Best K=33:')
    classifier = KNNClassification(training_file, test_file, k=33)
    classifier.retrieve_metrics()
