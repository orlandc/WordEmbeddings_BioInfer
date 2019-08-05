import os
import sys
import nltk
import multiprocessing
import itertools
from glove import Corpus, Glove

#nltk.download('punkt')

path = os.getcwd() + '/raw'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

sentences = []

for i, f in enumerate(files):
    archivo = open(f, "r")
    fl = archivo.readlines()
    for x in fl:
        tokens = nltk.word_tokenize(x)
        sentences.append(tokens)

#sentences = list(itertools.islice(Text8Corpus('text8'),None))

corpus = Corpus()
corpus.fit(sentences, window=5)

num_components = [20, 50, 100] 

for p in num_features:
        glove = Glove(no_components=100, learning_rate=0.05)
        glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        glove.save('model/glove_model_bioinfer_' + str(p) +  '_.txt')

        del glove