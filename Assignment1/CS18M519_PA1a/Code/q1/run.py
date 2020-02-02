import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from Code.q1.synthetic_data_creation import SyntheticDataGenerator

if __name__ == '__main__':
    data_generator = SyntheticDataGenerator(num_of_features=20, num_of_samples_per_class=2000, number_of_classes=2)
    data_generator.generate_data()
