import nltk
nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(nltk.corpus.stopwords.words('english'))

from sklearn.model_selection import train_test_split
import numpy
import re
import string
numpy.random.seed(42)

import sys

TEST_FRACTION = 0.3
DATASETS = {"bbcsport": {"train": "../../../data/bbcsport_train_test_merged",\
                         "test": None},\
            "r8": {"train": "../../../data/r8_train_merged",\
                     "test": "../../../data/r8_test_merged"},\
            "twitter": {"train": "../../../data/twitter_train_test_merged",\
                        "test": None},\
            "sst5": {"train": "../../../data/sst5_train",\
                     "test": "../../../data/sst5_test"},\
            "amazon": {"train": "../../../data/amazon_train_test_merged",\
                        "test": None},\
            "classic": {"train": "../../../data/classic_train_test_merged",\
                        "test": None},\
            "ohsumed": {"train": "../../../data/ohsumed_train_merged",\
                     "test": "../../../data/ohsumed_test_merged"}\
           }

DATASET_NAMES = list(DATASETS.keys())


def __preproc(sentence):
    sentence = sentence.strip().lower()
    sentence = re.sub(r"\d+", "", sentence)
    if sys.version_info[0] < 3:
        sentence = sentence.translate(string.maketrans("", ""), string.punctuation)
    else:
        sentence = sentence.translate(sentence.maketrans("", "", string.punctuation))
    sentence = [w for w in nltk.word_tokenize(sentence) if w not in stopwords]
    sentence = " ".join(sentence)
    return sentence


def __read_file(f, remove_stop = True, lower = True):
    data = []
    labels = []
    with open(f, "r") as openfile:
        for line in openfile:
            splitline = line.strip().strip("\n").split("\t")
            sentence = splitline[0]
            label = splitline[1]

            data.append(__preproc(sentence))
            labels.append(label)

    return data, labels


def read_dataset(name):
    train_path = DATASETS[name]["train"]
    test_path = DATASETS[name]["test"]

    X_train, Y_train = __read_file(train_path)
    X_test = None
    Y_test = None

    if test_path is not None:
        X_test, Y_test = __read_file(test_path)

        idx_train = numpy.random.permutation(len(X_train))
        idx_test = numpy.random.permutation(len(X_test))

        X_train = numpy.asarray(X_train)[idx_train].tolist()
        Y_train = numpy.asarray(Y_train)[idx_train].tolist()
        X_test = numpy.asarray(X_test)[idx_test].tolist()
        Y_test = numpy.asarray(Y_test)[idx_test].tolist()
    else:
        X_train, X_test, Y_train, Y_test = train_test_split\
        (X_train,\
         Y_train,\
         test_size = TEST_FRACTION,\
         shuffle = True,
         random_state = 42,
         stratify = Y_train\
        )

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    for name in DATASET_NAMES:
        X_train, Y_train, X_test, Y_test = read_dataset(name)

        print (name)
        print (X_train[0], Y_train[0])
        print (X_test[0], Y_test[0])
        print (len(X_train), len(Y_train))
        print (len(X_test), len(Y_test))
