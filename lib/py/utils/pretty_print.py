##### For KNN #####
foldername = "../../../evaluation-knn"

import os

stats = []
for filename in os.listdir(foldername):
	if "DS_Store" in filename:
		continue
	filepath = os.path.join(foldername, filename)

	acc = None
	p = None
	r = None
	fscore = None
	with open(filepath) as f:
		data = f.read().split("\n")

		for row in data:
			if len(row.split()) == 0:
				continue
			if row.split()[0] == "accuracy":
				acc = row.split()[1]
			if row.split()[0] == "weighted":
				p = row.split()[2]
				r = row.split()[3]
				fscore = row.split()[4]
	
	dataset = filename.split("-")[0]
	distancename = filename.split("-")[1].replace(".txt", "")
	stats.append({"dataset": dataset, "distance": distancename, "accuracy": acc, "P": p, "R": r, "F1": fscore})

import pandas

df = pandas.DataFrame(stats)
df = df.sort_values(["dataset", "distance"])
df.to_csv("../../../knn-results.tsv", sep = "\t", index = False)

##### For embeddings #####
foldername = "../../../evaluations"

stats = []
for filename in os.listdir(foldername):
	if "DS_Store" in filename:
		continue
	filepath = os.path.join(foldername, filename)
	
	acc = []
	p = []
	r = []
	fscore = []
	params = []
	with open(filepath) as f:
		data = f.read().split("\n")

		for row in data:
			if len(row.split()) == 0:
				continue
			if row.endswith(".p"):
				params.append(row.split()[2].replace(".p", ""))
			if row.split()[0] == "accuracy":
				acc.append(row.split()[1])
			if row.split()[0] == "weighted":
				p.append(row.split()[2])
				r.append(row.split()[3])
				fscore.append(row.split()[4])

	dataset = filename.split("-")[0]
	distancename = filename.split("-")[1].replace(".txt", "")
	for i in range(len(acc)):
		stats.append({"dataset": dataset, "distance": distancename, "accuracy": acc[i], "P": p[i], "R": r[i], "F1": fscore[i], "params": params[i]})


df = pandas.DataFrame(stats)
df = df.sort_values(["dataset", "distance"])

df.to_csv("../../../embedding-results.tsv", sep = "\t", index = False)
