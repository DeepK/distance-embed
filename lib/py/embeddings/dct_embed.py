# key in data name
import sys, os
name = sys.argv[1]

# load data
from py.utils.load_data import read_dataset

X_train, _, X_test, _ = read_dataset(name)

from py.utils.sent2vec import sent2vec
from py.utils.safe_pickle import pickle_dump
from tqdm import tqdm

from scipy.fftpack import dct
import numpy

def dct_embedding(c, vectors):
    if len(vectors) <= 1:
        return numpy.zeros(300 * c)
    sentvec = numpy.asarray(vectors)
    if sentvec.shape[0] < c:
        sentvec = numpy.reshape(
            dct(sentvec, n= c, norm='ortho', axis=0)[:c,:], (c*sentvec.shape[1],)
        )
    else:
        sentvec = numpy.reshape(
            dct(sentvec, norm='ortho', axis=0)[:c,:], (c*sentvec.shape[1],)
        )
    return sentvec

components = [1, 2, 3, 4, 5, 6]
dirname = "../../../exact_embeddings/dct_" + name
if not os.path.exists(dirname):
    os.mkdir(dirname)

for c in components:
    emb_mat = []
    for s in tqdm(X_train):
        emb_s = dct_embedding(c, sent2vec(s))
        emb_mat.append(emb_s)
    for s in tqdm(X_test):
        emb_s = dct_embedding(c, sent2vec(s))
        emb_mat.append(emb_s)
    emb_mat = numpy.asarray(emb_mat)
    pickle_dump(emb_mat, dirname + "/" + str(c) + ".p")