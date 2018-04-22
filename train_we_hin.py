from gensim.models import Word2Vec
import sys, gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#Supply as command line args
fname = sys.argv[1]


#If input is in a large file , use this!

class LoadFile(object):
    #Returns an iterator to iterate on...
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


sentences_ted = LoadFile(fname)
model_ted = Word2Vec(sentences_ted, size=100, window=5, min_count=4, workers=4,sg=1)
model_ted.save('./models/Hindi/hin')
