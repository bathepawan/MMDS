import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from Code.common.jaccard_similarity import JaccardSimilarity

import pandas as pd

if __name__ == '__main__':
    D = 39861
    W = 28102
    N = 6400000

    first_doc = int(input('> Please enter first document ID: '))
    second_doc = int(input('> Please enter second document ID: '))
    print('Calculating Jaccard Similarity, please wait')
    column_names = ['docID', 'wordID', 'word_count']
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                                  'Data', 'docword.enron.txt'), delimiter=' ', skiprows=3, header=None, names=column_names)
    doc_ids = range(1, 39862)
    document_sets = {}
    for doc_id in doc_ids:
        document_sets[doc_id] = set(df[df.docID == doc_id].wordID)
    js = JaccardSimilarity(document_sets)
    print('Calculated Jaccard similarity : {}'.format(js.calculate_jaccard_similarity(first_doc, second_doc)))

    for p in range(16, 1024, 32):
        print('Estimated Jaccard Similarity for rows {} is : {} '.format(p,
                                                                         js.estimate_jaccard_similarity_using_minhashing(first_doc, second_doc, permutations=p)))


