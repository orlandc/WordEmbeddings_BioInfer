import os
import sys
import nltk
import multiprocessing
from gensim.models import FastText

#nltk.download('punkt')

path = os.getcwd() + '/raw'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

sentences = []
diccionario = dict()

for i, f in enumerate(files):
    archivo = open(f, "r")
    fl = archivo.readlines()
    for x in fl:
        tokens = nltk.word_tokenize(x)
        sentences.append(tokens)
        for word in tokens:
            if word not in diccionario:
                diccionario[word] = 1
            else:
                diccionario[word] += 1

print ("numero de oraciones presentes en el corpus " + str(len(sentences)))
print ("numero de palabras unicas " + str(len(diccionario)))


num_features = [20, 50, 100]              #Dimensionality of the resulting word vectors
min_word_count = 1                        #Minimum word count threshold
num_workers = multiprocessing.cpu_count() #Number of threads to run in parallel
context_size = 5                          #Context window length
seed = 1                                  #Seed for the RNG, to make the result reproducible

for p in num_features:
    fasttext_model = FastText(
        sentences=sentences,
        size=p,
        window=context_size,
        min_count=min_word_count,
        workers=num_workers, 
        sg=1                              #skip-gram
    )

    fasttext_model.wv.save_word2vec_format('model/fasttext_skip-gram_model_bioinfer_' + str(p) +  '.txt', binary=False)

    del fasttext_model