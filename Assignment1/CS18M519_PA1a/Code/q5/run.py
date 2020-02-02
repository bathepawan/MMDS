import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir, cnc_headers
from Code.q5.linear_regressor import LinearRegressor

if __name__ == '__main__':
    dataset_file = os.path.join(dataset_dir, 'CandC-Imputed_mean.csv')

    linear_regression = LinearRegressor(dataset_file, cnc_headers,  'state', 'population', 'PolicBudgPerPop',
                                        'ViolentCrimesPerPop', 'ViolentCrimesPerPop', 5,  0.2)
    linear_regression.compute_rse()
