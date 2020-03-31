from py.utils.safe_pickle import pickle_load
from py.classification.classifier import train_best

import sys
dataset = sys.argv[1]
method = sys.argv[2]

# need to split into train-test
# to do this, need to load original and get the indices
# load data
from py.utils.load_data import read_dataset
X_train, Y_train, X_test, Y_test = read_dataset(dataset)
train_idx = len(X_train)

dirname = "../../../exact_embeddings/" + method + "_" + dataset
import subprocess

fnames = subprocess.Popen(["ls", dirname], stdout = subprocess.PIPE)
fnames = fnames.stdout.read()[:-1].decode().split("\n")

for fname in fnames:
	print (dataset, method, fname)
	fname = dirname + "/" + fname

	exact_embeddings = pickle_load(fname)

	train_emb = exact_embeddings[:train_idx]
	test_emb = exact_embeddings[train_idx:]

	report, acc = train_best(train_emb, Y_train, test_emb, Y_test)

	print (report)
	print (acc)