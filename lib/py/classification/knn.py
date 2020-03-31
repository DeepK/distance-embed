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

# need to split into train-test
# to do this, need to load original and get the indices
# load data
from py.utils.load_data import read_dataset
X_train, Y_train, X_test, Y_test = read_dataset(dataset)
train_idx = len(X_train)

X_train_mat = distmat[:train_idx, :train_idx]
X_test_train_mat = distmat[train_idx:, :train_idx]

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report, accuracy_score

knn = KNN(n_neighbors = 1, metric = "precomputed")

knn.fit(X_train_mat, Y_train)
predict = knn.predict(X_test_train_mat)

report = classification_report(Y_test, predict, digits = 5)
acc = accuracy_score(Y_test, predict)

print (report)
print (acc)