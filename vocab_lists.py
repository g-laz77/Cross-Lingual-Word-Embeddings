#################===README===########################################
#                                                                   #
#    Run system as python vocab_lists.py filename_you_want_vocab    #
#                                                                   #
#####################################################################

import sys

fname = sys.argv[1]

global_dict={}

class LoadFile(object):
    #Returns an iterator to iterate on...
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

sentences = LoadFile(fname)
for sentence in sentences:
    for token in sentence:
        token = token.lower()
        if token in global_dict:
            global_dict[token]+=1
        else:
            global_dict[token]=1


for key, value in sorted(global_dict.iteritems(), key=lambda(k,v): (v,k), reverse=True):
    print("%s: %s" % (key, value))