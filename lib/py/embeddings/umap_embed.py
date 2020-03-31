import sys
dataset = sys.argv[1]
distname = sys.argv[2]

from numpy import asarray, where, load, nan_to_num, max

distmat = None
if distname == "wmddist":
    import pickle

    with open("../../../produced/wmddist_" + dataset + ".pk", 'rb') as f:
        distmat = pickle.load(f, encoding="latin1")

    distmat = asarray(distmat)
    distmat = where(distmat, distmat, distmat.T)
    distmat = nan_to_num(distmat)
else:
    fname = "../../../produced/" + distname + "_" + dataset + ".numpyz.npz"

    from scipy.spatial.distance import squareform
    distmat = squareform(load(fname)["dist"])
    distmat = nan_to_num(distmat)

distmat[distmat<0]=0.0
distmat = distmat/max(distmat)
print (distmat.shape)

from umap import UMAP
from py.utils.safe_pickle import pickle_dump
import os

dirname = "../../../exact_embeddings/" + distname + "_" + dataset
if not os.path.exists(dirname):
    os.mkdir(dirname)

n_neighbors = 40
for n_components in [50, 100, 300, 1000]:
    for min_dist in [1.0, 1.5, 2.0]:
        for spread in [1.0, 2.5]:
            if min_dist > spread:
                continue

            print (n_components, n_neighbors, min_dist, spread)
            t = UMAP(n_components=n_components, n_neighbors = n_neighbors,\
                min_dist=min_dist, metric = "precomputed", random_state=42,\
                n_epochs = 1000, spread = spread)
            embeddings = t.fit_transform(distmat)
            print (embeddings.shape)
            pickle_dump(embeddings, dirname + "/" + str(n_components) + "-" + str(min_dist) + "-" + str(spread) + ".p")
