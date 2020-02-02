import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir, cnc_headers
from Code.q7.logistic_regressor import LogisticRegressor
from pandas import read_csv


if __name__=='__main__':
    DS2_train = read_csv(os.path.join(dataset_dir, 'DS2-train.csv'), sep='\s*,\s*',
                         header=None, encoding='ISO-8859-1', engine='python')

    DS2_test = read_csv(os.path.join(dataset_dir, 'DS2-test.csv'), sep='\s*,\s*',
                        header=None, encoding='ISO-8859-1', engine='python')

    logistic_regression_model = LogisticRegressor(DS2_train, DS2_test)
    print('Classification Report for Logistic Regression :\n')
    logistic_regression_model.train_and_print_metrics()

    l1_regularized_logistic_regression_model = LogisticRegressor(DS2_train, DS2_test, l1_regularized=True)
    print('Classification Report for L1 Regularized Logistic Regression :\n')
    l1_regularized_logistic_regression_model.train_and_print_metrics()
