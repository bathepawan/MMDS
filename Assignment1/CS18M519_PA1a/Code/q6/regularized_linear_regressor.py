import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir, cnc_headers

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np


class RegularizedLinearRegressor(object):
    def __init__(self, dataset_file, attributes, start_attribute, start_predictive_attribute, end_predictive_attribute,
                 end_attribute, output_attribute, number_of_runs=5, test_size=0.2, penalty=1):
        self.dataset = read_csv(dataset_file, sep=',', header=0, names=attributes, encoding='ISO-8859-1', engine='python')
        self.attributes = attributes
        self.start_attribute = start_attribute
        self.start_predictive_attribute = start_predictive_attribute
        self.end_predictive_attribute = end_predictive_attribute
        self.end_attribute = end_attribute
        self.output_attribute = output_attribute
        self.number_of_runs = number_of_runs
        self.test_size = test_size
        self.penalty = penalty
        self.rss = []
        self.reg_coefficients = []
        self.sorted_features_set = []
        self.imp_features = []

    def compute_rse(self):
        np.set_printoptions(suppress=True)

        for split in range(1, self.number_of_runs+1):
            x_train, x_test, y_train, y_test = train_test_split(
                self.dataset.loc[:, self.start_attribute: self.end_predictive_attribute],
                self.dataset.loc[:, self.output_attribute],
                test_size=self.test_size, random_state=split)
            training_data = x_train.copy()
            training_data[self.output_attribute] = y_train
            test_data = x_test.copy()
            test_data[self.output_attribute] = y_test

            test_file = os.path.join(dataset_dir, "CandC−train{}.csv".format(split))
            train_file = os.path.join(dataset_dir, "CandC−test{}.csv".format(split))
            training_data.to_csv(test_file, header=False, index=False, float_format='%.2f')
            test_data.to_csv(train_file, header=False, index=False, float_format='%.2f')
            # print('Split :{} , train file: {}, test file :{} '.format(split, train_file, test_file))

            linear_model = Ridge(alpha=self.penalty)
            linear_model.fit(training_data.loc[:, self.start_predictive_attribute: self.end_predictive_attribute],
                             training_data.loc[:, self.output_attribute])
            # print('Coefficients for split {} are {}'.format(split, linear_model.coef_))
            y_predicted = linear_model.predict(test_data.loc[:, self.start_predictive_attribute:
                                                                self.end_predictive_attribute])
            rss_current = ((y_predicted - y_test) ** 2).sum()
            self.store_features(coefs=linear_model.coef_, names=self.attributes[5:-1], max_features=50)
            # print('RSS for split {} is {}'.format(split, round(rss_current, 3)))
            self.rss.append(rss_current)
            self.reg_coefficients.append((linear_model.coef_))
        print('*'*20+'Alpha {}:'.format(self.penalty)+'*'*20)
        print('\n RSS for 5 splits is {}'.format(self.rss))
        print('\n Lowest RSS for best fit is {} for split number {}'
              .format(min(self.rss), self.rss.index(min(self.rss))+1))
        print('\n Coefficients learned for this best fit are : \n {} '
              .format(self.reg_coefficients[self.rss.index(min(self.rss))]))

        np.savetxt('coeffs-lamda{}.csv'.format(self.penalty), self.reg_coefficients[self.rss.index(min(self.rss))],
                   delimiter=",")
        '''
        print('\n Equation of fitted line is : \n')
        self.print_model(coefs=self.reg_coefficients[self.rss.index(min(self.rss))],
                         names=self.attributes[5:-1], output=self.output_attribute)
        '''
        print('\n Average RSS over {} splits is {} '.format(self.number_of_runs,
                                                            round(sum(self.rss)/len(self.rss), 3),))

    def print_model(self, coefs, names=None, output='Y'):
        feature_coef_map = zip(coefs, names)
        feature_coef_map = sorted(feature_coef_map, key=lambda x: -np.abs(x[0]))
        equation = " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in feature_coef_map)
        print('{} = {}'.format(output, equation))

    def store_features(self, coefs, names=None, max_features=50):
        feature_coef_map = zip(coefs, names)
        feature_coef_map = sorted(feature_coef_map, key=lambda x: -np.abs(x[0]))
        scoef, s_names = zip(*feature_coef_map)
        self.sorted_features_set.append(s_names[0:max_features])

    def retrive_features(self, ):
        list_of_features_set = [set(feature_set) for feature_set in self.sorted_features_set]
        return set.intersection(*list_of_features_set)


if __name__ == '__main__':
    dataset_file = os.path.join(dataset_dir, 'CandC-Imputed_mean.csv')

    for lam in range(1, 5):
        linear_regression = RegularizedLinearRegressor(dataset_file, cnc_headers,
                                                       'state', 'population', 'PolicBudgPerPop', 'ViolentCrimesPerPop',
                                                       'ViolentCrimesPerPop', 5,  0.2, penalty=lam)
        linear_regression.compute_rse()
