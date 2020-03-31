# key in data name
import sys, os
name = sys.argv[1]

# load data
from py.utils.load_data import read_dataset

X_train, _, X_test, _ = read_dataset(name)

from py.utils.sent2vec import sent2vec
from py.utils.safe_pickle import pickle_dump
from tqdm import tqdm

from pydmd import HODMD
import numpy

def sentence_to_dmd_vec(to_keep, time_lags, vectors):
    min_len = to_keep + max(time_lags)
    if len(vectors) <= min_len:
        return numpy.zeros(to_keep * len(time_lags) * 300)
    v = numpy.asarray(vectors)
    list_of_modes = []
    for d in time_lags:
        dmd = HODMD(svd_rank = to_keep, opt=True, exact=True, d=d)
        dmd.fit(v.T)
        fmode = dmd.modes.T
        list_of_modes.append(numpy.hstack(numpy.absolute(fmode)))

    mean_vec = numpy.power(numpy.mean(numpy.power(numpy.array(vectors, dtype=complex), 1.0), axis=0), 1 / 1.0).real
    list_of_modes.append(mean_vec)

    return numpy.hstack(list_of_modes)

components = [1, 2, 3]
time_lags = [[1], [2], [3], [1, 2], [1, 2, 3], [1, 2, 3, 4]]
dirname = "../../../exact_embeddings/eigensent_" + name
if not os.path.exists(dirname):
    os.mkdir(dirname)

for c in components:
    for tl in time_lags:
        emb_mat = []
        for s in tqdm(X_train):
            emb_s = sentence_to_dmd_vec(c, tl, sent2vec(s))
            emb_mat.append(emb_s)
        for s in tqdm(X_test):
            emb_s = sentence_to_dmd_vec(c, tl, sent2vec(s))
            emb_mat.append(emb_s)
        emb_mat = numpy.asarray(emb_mat)

        fname = str(c) + "_" + "_".join([str(e) for e in tl])
        pickle_dump(emb_mat, dirname + "/" + fname + ".p")
