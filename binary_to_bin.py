import sys
from gensim.models import KeyedVectors,Word2Vec

fname = sys.argv[1]
model = Word2Vec.load(fname)
# from gensim.models.keyedvectors import KeyedVectors

# model = KeyedVectors.load_word2vec_format(fname, binary=True)
model.wv.save_word2vec_format(fname+".bin", binary=False)

