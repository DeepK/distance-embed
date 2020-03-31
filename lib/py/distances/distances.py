from scipy.spatial.distance import directed_hausdorff
from dcor import energy_distance

from py.utils.sent2vec import MODEL, sent2vec

class PairedDistance(object):
    def __init__(self, s1, s2):
        self.s1_text = s1
        self.s2_text = s2
        self.s1 = sent2vec(s1)
        self.s2 = sent2vec(s2)


    def hausdorff(self):
        try:
            return max(directed_hausdorff(self.s1, self.s2, seed=42)[0],\
                       directed_hausdorff(self.s2, self.s1, seed=42)[0])
        except:
            return 0.0


    def wmd(self):
        try:
            return MODEL.wmdistance(self.s1_text, self.s2_text)
        except:
            return 0.0

    def energy_dist(self):
        try:
            return energy_distance(self.s1, self.s2)
        except:
            return 0.0


if __name__ == "__main__":
    sentences = ["Obama is talking to the press in chicago", "Obama is speaking in illinois", "Putin lives in Moscow, Russia"]
    pair_1 = PairedDistance(sentences[0], sentences[1])
    pair_2 = PairedDistance(sentences[0], sentences[2])

    print ("Distances between '{}' and '{}' are:".format(sentences[0], sentences[1]))
    print ("energy: ", pair_1.energy_dist())
    print ("hausdorff: ", pair_1.hausdorff())
    print ("wmd: ", pair_1.wmd())

    print ()

    print ("Distances between '{}' and '{}' are:".format(sentences[0], sentences[2]))
    print ("energy: ", pair_2.energy_dist())
    print ("hausdorff: ", pair_2.hausdorff())
    print ("wmd: ", pair_2.wmd())
