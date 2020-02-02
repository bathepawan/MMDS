from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

dataset_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, 'Dataset')


class LogisticRegressor(object):
    def __init__(self, train_data, test_data, l1_regularized=False):
        self.train_data = train_data
        self.test_data = test_data
        self.l1_regularized = l1_regularized
        if l1_regularized:
            self.logistic_regression_model = LogisticRegression(penalty='l1', solver='saga', warm_start=True)
        else:
            self.logistic_regression_model = LogisticRegression(multi_class='ovr', solver='saga', max_iter=200)

    def train_and_print_metrics(self):
        self.logistic_regression_model.fit(self.train_data.loc[:, 0:95], self.train_data.loc[:, 96])
        y_predicted = self.logistic_regression_model.predict(self.test_data.loc[:, 0:95])
        print(metrics.classification_report(y_predicted, self.test_data.loc[:, 96]))


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
