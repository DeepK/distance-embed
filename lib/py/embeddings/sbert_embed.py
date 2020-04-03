# key in data name
import sys, os
name = sys.argv[1]

# load data
from py.utils.load_data import read_dataset

X_train, _, X_test, _ = read_dataset(name)

from py.utils.safe_pickle import pickle_dump
from tqdm import tqdm
import numpy
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')

dirname = "../../../exact_embeddings/sbert_" + name
if not os.path.exists(dirname):
    os.mkdir(dirname)

emb_mat = []
for s in tqdm(X_train):
    emb_mat.append(model.encode([s.lower()])[0])
for s in tqdm(X_test):
    emb_mat.append(model.encode([s.lower()])[0])
emb_mat = numpy.asarray(emb_mat)

pickle_dump(emb_mat, dirname + "/" + "emb.p")
