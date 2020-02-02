import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.q2.linear_classification import LinearClassification
from Code.common.config import dataset_dir

if __name__ == '__main__':
    training_file = os.path.join(dataset_dir, 'DS1-train.csv')
    test_file = os.path.join(dataset_dir, 'DS1-test.csv')
    classifier = LinearClassification(training_file, test_file)
    classifier.retrieve_metrics()
