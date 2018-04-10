import numpy as np
import sys


from sklearn.utils import check_random_state

class WordEmbeddings(object):
	def __init__(self):
		self.num_words = 0
		self.total_count = 0
		self.words = []
		self.embedding_dim = 0
		self.vectors = np.zeros((0, 0))
		self.counts = np.zeros(0, dtype=int)
        self.probs = np.zeros(0)
        self.word_dict = dict([])

	def load_from_word2vec(self, file_prefix):
		vocab_file = file_prefix + '.vocab'
		vec_file = file_prefix + '.w2v'
                
		vec_fs = open(vec_file)
		line = vec_fs.readline()
		tokens = line.split()
		self.counts = np.zeros(self.num_words, dtype=int)                       # count of each word in vocab
		self.num_words = int(tokens[0])                                         # number of unique words( size of vocab)
		self.embedding_dim = int(tokens[1])                                     # embedding dimensions (typically 300)
		self.vectors = np.zeros((self.num_words, self.embedding_dim))           # Embedding matrix
		self.probs = np.ones(self.num_words)                                    # word probabilities
		for i, line in enumerate(vec_fs):
			tokens = line.split()
			word = tokens[0]
			self.words.append(word)
			self.word_dict[word] = i
			self.vectors[i] = [float(x) for x in tokens[1:]]

		vocab_fs = open(vocab_file)
		for line in vocab_fs:
			tokens = line.split()
			word, count = tokens[0], int(tokens[1])
			self.counts[self.word_dict[word]] = count

		self.total_count = self.counts.sum()                                # total number of words
		self.probs = self.probs * self.counts
		self.probs = self.probs / self.total_count   # calculating all probabilities

	def downsample_frequent_words(self, frequency_threshold = 1e-3):
		threshold_count = float(frequency_threshold * self.total_count)
		self.probs = (np.sqrt(self.counts / threshold_count) + 1) * (threshold_count / self.counts)
		self.probs = np.maximum(self.probs, 1.0)
		self.probs *= self.counts
		self.probs /= self.probs.sum()

	def sample_batches(self, batch_size = 1, train_set_ids = None, random_state = 0):
		rng = check_random_state(random_state)
		if not train_set_ids == None:
			p = self.probs[train_set_ids]
			p /= p.sum()
			a = train_set_ids            
		else:
			p = self.probs            
			a = self.num_words
		while 1:
			rv = rng.choice(a, size=batch_size, replace=True, p=p)
			yield rv
