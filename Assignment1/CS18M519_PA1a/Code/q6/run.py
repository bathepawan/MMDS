import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir, cnc_headers
from Code.q6.regularized_linear_regressor import RegularizedLinearRegressor
from Code.q6.reduced_linear_regressor import ReducedLinearRegressor
from pandas import read_csv


if __name__ == '__main__':
    dataset_file = os.path.join(dataset_dir, 'CandC-Imputed_mean.csv')

    for lam in range(1, 5):
        linear_regression = RegularizedLinearRegressor(dataset_file, cnc_headers,
                                                       'state', 'population', 'PolicBudgPerPop', 'ViolentCrimesPerPop',
                                                       'ViolentCrimesPerPop', 5,  0.2, penalty=lam)
        linear_regression.compute_rse()

    print('By experimentation it is observed that model wiht lambda value of 1 gives best fit. ')
    print('Use this value to retrieve top features and train model on small set of features: ')
    linear_regression = RegularizedLinearRegressor(dataset_file, cnc_headers,
                                                   'state', 'population', 'PolicBudgPerPop', 'ViolentCrimesPerPop',
                                                   'ViolentCrimesPerPop', 5, 0.2, penalty=1)
    linear_regression.compute_rse()
    important_features = linear_regression.retrive_features()

    dataset = read_csv(dataset_file, sep=',', header=0, names=cnc_headers, encoding='ISO-8859-1', engine='python')

    print('Features retrieved using regularized linear regression {}'.format(important_features))

    reduced_linear_regression = ReducedLinearRegressor(dataset=dataset, features=important_features,
                                                       output='ViolentCrimesPerPop', number_of_runs=5,  test_size=0.2)
    reduced_linear_regression.compute_rse()
