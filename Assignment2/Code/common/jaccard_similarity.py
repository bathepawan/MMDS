from datasketch import MinHash
from Code.common.helper import get_word

class JaccardSimilarity(object):
    def __init__(self, documents_sets):
        self.documents = documents_sets

    def calculate_jaccard_similarity(self, first_doc, second_doc):
        s1 = self.documents.get(first_doc)
        s2 = self.documents.get(second_doc)
        return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))

    def estimate_jaccard_similarity_using_minhashing(self, first_doc, second_doc, permutations=128):
        h1 = MinHash(num_perm=permutations)
        h2 = MinHash(num_perm=permutations)
        for word in self.documents.get(first_doc):
            h1.update(get_word(word_id=word).encode('utf-8'))
        for word in self.documents.get(second_doc):
            h2.update(get_word(word_id=word).encode('utf-8'))
        return h1.jaccard(h2)


