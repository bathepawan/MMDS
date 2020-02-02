import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))

from Code.common.jaccard_similarity import JaccardSimilarity
import random
import pandas as pd

random.seed(9999)

if __name__ == '__main__':
    D = 39861
    W = 28102
    N = 6400000

    column_names = ['docID', 'wordID', 'word_count']
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                                  'Data', 'docword.enron.txt'), delimiter=' ', skiprows=3, header=None, names=column_names)
    doc_ids = range(1, 39862)
    document_sets = {}
    print('Preparing shringles sets, please wait')
    for doc_id in doc_ids:
        document_sets[doc_id] = set(df[df.docID == doc_id].wordID)
    js = JaccardSimilarity(document_sets)

    for attempt in range(1, 5):
        first_doc = random.randint(1, D)
        second_doc = random.randint(1, D)
        print('Jaccard Similarities for {} and {}'.format(first_doc, second_doc))
        print('Calculated Jaccard similarity : {}'.format(js.calculate_jaccard_similarity(first_doc, second_doc)))
        print('Estimated Jaccard Similarity is : {} '.format(js.estimate_jaccard_similarity_using_minhashing(first_doc,
                                                                                                         second_doc)))


