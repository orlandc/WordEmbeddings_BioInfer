import pickle
                     
with open('model/glove_model_bioinfer_50.bin', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    p = u.load()

keys=p.keys()
values=p.values()

with open("model/glove_model_bioinfer_50.txt", "w") as txt_file:
    for t, k in enumerate(p["word_vectors"]):
        for word, key in p["dictionary"].items():
            if (t == key):
                txt_file.write(word + " " + " ".join(str(elem) for elem in k) + "\n")
                break