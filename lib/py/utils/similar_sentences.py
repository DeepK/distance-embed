dataset = "amazon"
distname = "wmddist"
best_params = "100-1.0-1.0"

from py.utils.safe_pickle import pickle_load
embeddings = pickle_load("../../../exact_embeddings/" + distname + "_" + dataset + "/" + best_params + ".p")
print(embeddings.shape)

from py.utils.load_data import read_dataset
X_train, _, X_test, _ = read_dataset(dataset)
all_sent = X_train + X_test
print(len(all_sent))

from scipy.spatial.distance import cosine
def sim_sent(embeddings, query_idx):
    dist = 1
    best_match_idx = None
    query_emb = embeddings[query_idx]
    for i in range(len(embeddings)):
        if i != idx:
            d = cosine(query_emb, embeddings[i])
            if d < dist:
                dist = d
                best_match_idx = i
    return 1-dist, best_match_idx


random_indices = [20, 1000, 3000, 6000]
for idx in random_indices:
    print(idx, "Sentence: ", all_sent[idx])
    sim, best_idx = sim_sent(embeddings, idx)
    print (sim, "Best match: ", all_sent[best_idx])
