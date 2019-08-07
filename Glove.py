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

corpus = Corpus()
corpus.fit(sentences, window=5)

num_components = [20, 50, 100]
no_threads = multiprocessing.cpu_count()

for p in num_components:
    glove = Glove(no_components=p, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=30, no_threads=no_threads, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    #glove.save('model/glove_model_bioinfer_' + str(p) +  '.bin')
    #print(dir(glove))
    with open('model/glove_model_bioinfer_' + str(p) +  '.txt', "w") as txt_file:
        for t, k in enumerate(glove.word_vectors):
            for word, key in glove.dictionary.items():
                if (t == key):
                    txt_file.write(word + " " + " ".join(str(elem) for elem in k) + "\n")
                    break
    del glove