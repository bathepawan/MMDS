import pandas as pd
import sys
import os
import random
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from Code.common.KNearestNeighbors import KNearestNeighbors

random.seed(1000)

if __name__ == '__main__':
    D = 39861
    W = 28102
    N = 6400000

    column_names = ['docID', 'wordID', 'word_count']
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir,  os.pardir, 'Data', 'docword.enron.txt'), delimiter=' ', skiprows=3, header=None, names=column_names)
    doc_ids = range(1, 39862)
    document_sets = {}
    print('Forming sets of documents from shringles... ')
    for doc_id in doc_ids:
        document_sets[doc_id] = set(df[df.docID == doc_id].wordID)

    k = int(input('> Please enter value of K: '))
    knn = KNearestNeighbors(document_sets)

    doc_id_key = int(input('> Please enter doc ID: '))
    knn.find_k_nearest_neighbors_using_brute_force(doc_id=doc_id_key, k=k)
    knn.find_k_nearest_neighbors_using_lhs(doc_id=doc_id_key, k=k)
