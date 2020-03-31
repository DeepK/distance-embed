# key in data name
import sys, os
name = sys.argv[1]

# load data
from py.utils.load_data import read_dataset

X_train, _, X_test, _ = read_dataset(name)

from py.utils.sent2vec import sent2vec
from py.utils.safe_pickle import pickle_dump
from tqdm import tqdm

# pmean calculator
import numpy
def p_mean_vector(powers, vectors):
    if len(vectors) <= 1:
        return numpy.zeros(300 * len(powers))
    embeddings = []
    for p in powers:
        embeddings.append(
            numpy.power(numpy.mean(numpy.power(numpy.array(vectors, dtype=complex), p), axis=0), 1 / p).real)
    return numpy.hstack(embeddings)


powers_list = [[1.0], [1.0, 2.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
dirname = "../../../exact_embeddings/powermeans_" + name
if not os.path.exists(dirname):
    os.mkdir(dirname)

for powers in powers_list:
    emb_mat = []
    for s in tqdm(X_train):
        emb_s = p_mean_vector(powers, sent2vec(s))
        emb_mat.append(emb_s)
    for s in tqdm(X_test):
        emb_s = p_mean_vector(powers, sent2vec(s))
        emb_mat.append(emb_s)
    emb_mat = numpy.asarray(emb_mat)

    powers_name = "_".join([str(e) for e in powers])
    pickle_dump(emb_mat, dirname + "/" + powers_name + ".p")
