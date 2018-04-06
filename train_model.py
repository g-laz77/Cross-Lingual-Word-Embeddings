from embed_funcs import WordEmbeddings
import sys
import numpy as np

def cosine_similarity(a_matrix, b_matrix):
    return ((a_matrix * b_matrix).sum(axis=1)) / ((a_matrix.norm(2, axis = 1))*(b_matrix.norm(2, axis = 1)))

print('Loading Hindi embeddings...')
we_hi = WordEmbeddings()
we_hi.load_from_word2vec('./hindi embeddings file here')
we_hi.downsample_frequent_words()
skn_hi = StandardScaler()
we_hi.vectors = skn_hi.fit_transform(we_hi.vectors).astype(theano.config.floatX)
we_batches_hi = we_hi.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)

print >> sys.stderr, 'Loading English embeddings...'
we_en = WordEmbeddings()
we_en.load_from_word2vec('./english embeddings file here')
we_en.downsample_frequent_words()
skn_en = StandardScaler()
we_en.vectors = skn_en.fit_transform(we_en.vectors).astype(theano.config.floatX)
we_batches_en = we_en.sample_batches(batch_size=HALF_BATCH_SIZE, random_state=rng)