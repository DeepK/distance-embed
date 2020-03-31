import warnings
warnings.filterwarnings("ignore")

# key in data name
import sys
name = sys.argv[1]

# load data
from py.utils.load_data import read_dataset
X_train, _, X_test, _ = read_dataset(name)
data = []
data.extend(X_train)
data.extend(X_test)
print ("Loaded {} sentences".format(len(data)))

# get distance measures
from py.distances.distances import PairedDistance

# calculate distances in parallel
from multiprocessing import Pool
import numpy
from time import time as ts

n_processes = 3
n = len(data)
k_max = n * (n - 1) // 2
k_step = n ** 2 // 100000 

hausdorffdist = numpy.zeros(k_max)
energydist = numpy.zeros(k_max)

def proc(start):
    hausdorffdist = []
    energydist = []
    k1 = start
    k2 = min(start + k_step, k_max)
    for k in range(k1, k2):
        i = int(n - 2 - int(numpy.sqrt(-8 * k + 4 * n * (n - 1) - 7) / 2.0 - 0.5))
        j = int(k + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2)
        
        a = data[i]
        b = data[j]
        paired_dist = PairedDistance(a, b)

        # get various distances
        energydist.append(paired_dist.energy_dist())
        hausdorffdist.append(paired_dist.hausdorff())

    return k1, k2, hausdorffdist, energydist


ts_start = ts()
with Pool(n_processes) as pool:
    for k1, k2, res1, res2 in pool.imap_unordered(proc, range(0, k_max, k_step)):
        hausdorffdist[k1:k2] = res1
        energydist[k1:k2] = res2
        print("{:.0f} minutes, {:,}..{:,} out of {:,}".format(
            (ts() - ts_start)/60, k1, k2, k_max))


print("Elapsed %.0f minutes" % ((ts() - ts_start) / 60))
print("Saving...")
numpy.savez("../../../produced/hausdorffdist_{}.numpyz".format(name), dist=hausdorffdist)
numpy.savez("../../../produced/energydist_{}.numpyz".format(name), dist=energydist)
print("DONE")
