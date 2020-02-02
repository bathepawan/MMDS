import os
import pandas as pd


def get_word(word_id):
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  os.pardir, os.pardir, 'Data', 'vocab.enron.txt'), delimiter=' ', names=['word'])
    return df.iloc[word_id-1,0]



