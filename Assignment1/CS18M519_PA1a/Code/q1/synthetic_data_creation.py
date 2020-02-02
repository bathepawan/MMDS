import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
import numpy as np
from sklearn.model_selection import train_test_split
from Code.common.config import dataset_dir


class SyntheticDataGenerator(object):
    def __init__(self, num_of_features, num_of_samples_per_class, number_of_classes=2):
        np.random.seed(0)
        self.num_of_features = num_of_features
        self.num_of_samples_per_class = num_of_samples_per_class
        self.num_of_classes = number_of_classes
        self.class_means = dict()
        self.class_covariance_matrix = None
        self.combined_dataset = None
        self.class_samples = dict()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.training_data = None
        self.test_data = None
        self.class_labels = None

    def get_random_mean(self, num):
        return np.random.rand(num)

    def get_random_covariance(self, num):
        A = np.random.rand(num, num)
        np.dot(A, A.transpose())
        return A

    def generate_data(self):
        self.class_covariance_matrix = self.get_random_covariance(self.num_of_features)
        for class_index in range(0, self.num_of_classes):
            np.random.seed(class_index)
            self.class_means[class_index] = self.get_random_mean(self.num_of_features)
            self.class_samples[class_index] = np.random.multivariate_normal(self.class_means[class_index],
                                                                            self.class_covariance_matrix,
                                                                            self.num_of_samples_per_class)

        samples = [value for key, value in self.class_samples.items()]
        self.combined_dataset = np.concatenate(samples).round(decimals=3)

        # Create Y variable which will hold class labels, since first half of combined dataset has
        # samples with class 0 and remaning has sample with class 1

        self.class_labels = np.array([0 if index < self.num_of_samples_per_class
                                      else 1 for index in range(0, self.num_of_samples_per_class * 2)])

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.combined_dataset,
                                                                                self.class_labels,
                                                                                test_size=0.30,
                                                                                random_state=2)

        self.y_train = self.y_train.reshape(len(self.y_train), 1)
        self.y_test = self.y_test.reshape(len(self.y_test), 1)

        self.training_data = np.append(self.x_train, self.y_train, axis=1)
        self.test_data = np.append(self.x_test, self.y_test, axis=1)
        np.savetxt(os.path.join(dataset_dir, "DS1-train.csv"), self.training_data, delimiter=",", fmt='%.3f')
        np.savetxt(os.path.join(dataset_dir, "DS1-test.csv"), self.test_data.round(decimals=3), delimiter=",", fmt='%.3f')
        print('Saved generated dataset at {}'.format(dataset_dir))


if __name__ == '__main__':
    data_generator = SyntheticDataGenerator(num_of_features=20, num_of_samples_per_class=2000, number_of_classes=2)
    data_generator.generate_data()
