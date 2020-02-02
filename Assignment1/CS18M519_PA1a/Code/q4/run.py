import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.q4.data_imputer import DataImputater
from Code.common.config import dataset_dir, cnc_headers
from pandas import read_csv


if __name__=='__main__':

    predictive_attributes = cnc_headers[5:-1]
    cnc_data_file = os.path.join(dataset_dir, 'communities.data')
    cnc_data = read_csv(cnc_data_file, sep='\s*,\s*', header=None, encoding='ISO-8859-1',
                        engine='python', names=[head.strip() for head in cnc_headers], na_values=['?'])
    for strategy in ['mean', 'interpolate']:
        data_imputer = DataImputater(cnc_data, strategy, predictive_attributes)
        data_imputer.impute()