import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.common.config import dataset_dir, cnc_headers

from pandas import read_csv


class DataImputater(object):
    def __init__(self, dataset, strategy, attributes_to_impute):
        self.dataset = dataset
        self.strategy = strategy
        self.attributes_to_impute = attributes_to_impute

    def impute(self):
        for attribute in self.attributes_to_impute:
            if self.strategy == 'mean':
                self.dataset[attribute].fillna(self.dataset[attribute].mean(), inplace=True)
            elif self.strategy == 'interpolate':
                self.dataset[attribute].fillna(self.dataset[attribute].interpolate(), inplace=True)
            else:
                self.dataset[attribute].fillna(self.dataset[attribute].mean(), inplace=True)

        filename= os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                               'Dataset', "CandC-Imputed_{}.csv".format(self.strategy))

        self.dataset.to_csv(filename, index=False)
        print('Imputed dataset saved to: {}.'.format(filename))
        return filename


if __name__=='__main__':
    predictive_attributes = cnc_headers[5:-1]
    cnc_data_file = os.path.join(dataset_dir, 'communities.data')
    cnc_data = read_csv(cnc_data_file, sep='\s*,\s*', header=None, encoding='ISO-8859-1',
                        engine='python', names=[head.strip() for head in cnc_headers], na_values=['?'])

    for strategy in ['mean', 'interpolate']:
        data_imputer = DataImputater(cnc_data, strategy, predictive_attributes)
        data_imputer.impute()