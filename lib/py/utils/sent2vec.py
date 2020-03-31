# load w2v model and prepare sentence to vectors
import gensim
import nltk
nltk.download('punkt')

MODEL = gensim.models.KeyedVectors.load_word2vec_format("../../../resources/GoogleNews-vectors-negative300.bin.gz", binary= True)
MODEL.init_sims(replace=True)

def sent2vec(sentence):
    return [MODEL[w] for w in nltk.word_tokenize(sentence) if w in MODEL]