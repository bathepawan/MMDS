from datasketch import MinHashLSH, MinHash
from Code.common.helper import get_word


class KNearestNeighbors(object):
    def __init__(self, documents_sets):
        self.documents = documents_sets
        self.similarities = {}

    def find_k_nearest_neighbors_using_brute_force(self, doc_id, k):
        self.similarities = {}
        s1 = self.documents.get(doc_id)
        for other_document in self.documents:
            if doc_id == other_document:
                self.similarities[doc_id] = 1
            else:
                # print('Calculate jaccard similarities between {} {} '.format(doc_id, other_document))
                s2 = self.documents.get(other_document)
                self.similarities[other_document] =\
                    float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
            # print('Jaccard similarity between {} and {} is {} '.format(doc_id, other_document,
            #                                                           self.similarities[other_document]))
        sorted_similarities = sorted(self.similarities.items(), key=lambda kv: kv[1], reverse=True)
        print('{} nearest neighbors (<neighbor , simliarity> tuple) of {} using brute force approach '.format(k, doc_id))
        print(sorted_similarities[0:k])
        return sorted_similarities[0:k]

    def find_k_nearest_neighbors_using_lhs(self, doc_id, k, lsh_threashhold=0.0001, num_perm=100, b=20, r=5):
        min_hash_dict = {}
        for doc in self.documents.keys():
            h = MinHash(num_perm=num_perm)
            for word in self.documents.get(doc):
                h.update(get_word(word_id=word).encode('utf-8'))
            min_hash_dict[doc] = h
        lsh = MinHashLSH(threshold=lsh_threashhold, num_perm=num_perm, params=(b, r))
        for doc, minhash in min_hash_dict.items():
            lsh.insert(doc, minhash)
        similar_elements = lsh.query(min_hash_dict.get(doc_id))
        print('{} nearest neighbors of {} using lhs '.format(k, doc_id))
        print(similar_elements)
        return similar_elements[0:k]



