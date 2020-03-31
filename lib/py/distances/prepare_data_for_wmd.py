# key in data name
import sys
name = sys.argv[1]

# load data and write in a suitable format for wmd code
from py.utils.load_data import read_dataset
X_train, Y_train, X_test, Y_test = read_dataset(name)
with open("../../../data/" + name + "_for_wmd.txt", "w") as f:
    for i in range(len(X_train)):
        f.write("{}\t{}\n".format(Y_train[i], X_train[i]))
    for i in range(len(X_test)):
        f.write("{}\t{}\n".format(Y_test[i], X_test[i]))

from py.distances.wmd.get_word_vectors import read_line_by_line
import gensim
# load word2vec model (trained on Google News)
model = gensim.models.KeyedVectors.load_word2vec_format('../../../resources/GoogleNews-vectors-negative300.bin.gz', binary=True)
vec_size = 300

# specify train/test datasets
train_dataset = "../../../data/" + name + "_for_wmd.txt" # e.g.: 'twitter.txt'
save_file = "../../../produced/" + name + "_vec.pk" # e.g.: 'twitter.pk'

# read document data
(X,BOW_X,y,C,words)  = read_line_by_line(train_dataset,[],model,vec_size)

# save pickle of extracted variables
import pickle
with open(save_file, 'wb') as f:
    pickle.dump([X, BOW_X, y, C, words], f)