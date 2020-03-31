set -x
alias python=python3

cd ../py/classification
#python knn.py bbcsport energydist >> ../../../evaluation-knn/bbcsport-energydist.txt
#python knn.py bbcsport wmddist >> ../../../evaluation-knn/bbcsport-wmddist.txt
#python knn.py bbcsport hausdorffdist >> ../../../evaluation-knn/bbcsport-hausdorffdist.txt
#python knn.py twitter wmddist >> ../../../evaluation-knn/twitter-wmddist.txt
#python knn.py twitter energydist >> ../../../evaluation-knn/twitter-energydist.txt
#python knn.py twitter hausdorffdist >> ../../../evaluation-knn/twitter-hausdorffdist.txt
#python knn.py r8 wmddist >> ../../../evaluation-knn/r8-wmddist.txt
#python knn.py r8 energydist >> ../../../evaluation-knn/r8-energydist.txt
#python knn.py r8 hausdorffdist >> ../../../evaluation-knn/r8-hausdorffdist.txt
#python knn.py sst5 wmddist >> ../../../evaluation-knn/sst5-wmddist.txt
#python knn.py sst5 energydist >> ../../../evaluation-knn/sst5-energydist.txt
#python knn.py sst5 hausdorffdist >> ../../../evaluation-knn/sst5-hausdorffdist.txt
#python knn.py 20ng wmddist >> ../../../evaluation-knn/20ng-wmddist.txt
#python knn.py 20ng energydist >> ../../../evaluation-knn/20ng-energydist.txt
#python knn.py 20ng hausdorffdist >> ../../../evaluation-knn/20ng-hausdorffdist.txt
#python knn.py amazon hausdorffdist >> ../../../evaluation-knn/amazon-hausdorffdist.txt
#python knn.py amazon energydist >> ../../../evaluation-knn/amazon-energydist.txt
#python knn.py amazon wmddist >> ../../../evaluation-knn/amazon-wmddist.txt
#python knn.py classic hausdorffdist >> ../../../evaluation-knn/classic-hausdorffdist.txt
#python knn.py classic energydist >> ../../../evaluation-knn/classic-energydist.txt
#python knn.py classic wmddist >> ../../../evaluation-knn/classic-wmddist.txt
#python knn.py ohsumed hausdorffdist >> ../../../evaluation-knn/ohsumed-hausdorffdist.txt
#python knn.py ohsumed energydist >> ../../../evaluation-knn/ohsumed-energydist.txt
#python knn.py ohsumed wmddist >> ../../../evaluation-knn/ohsumed-wmddist.txt

#python evaluate.py bbcsport energydist >> ../../../evaluations/bbcsport-energydist.txt
#python evaluate.py bbcsport wmddist >> ../../../evaluations/bbcsport-wmddist.txt
#python evaluate.py bbcsport hausdorffdist >> ../../../evaluations/bbcsport-hausdorffdist.txt
#python evaluate.py twitter wmddist >> ../../../evaluations/twitter-wmddist.txt
#python evaluate.py twitter energydist >> ../../../evaluations/twitter-energydist.txt
#python evaluate.py twitter hausdorffdist >> ../../../evaluations/twitter-hausdorffdist.txt
#python evaluate.py r8 wmddist >> ../../../evaluations/r8-wmddist.txt
#python evaluate.py r8 energydist >> ../../../evaluations/r8-energydist.txt
#python evaluate.py r8 hausdorffdist >> ../../../evaluations/r8-hausdorffdist.txt
#python evaluate.py sst5 wmddist >> ../../../evaluations/sst5-wmddist.txt
#python evaluate.py sst5 energydist >> ../../../evaluations/sst5-energydist.txt
#python evaluate.py sst5 hausdorffdist >> ../../../evaluations/sst5-hausdorffdist.txt
#python evaluate.py 20ng wmddist >> ../../../evaluations/20ng-wmddist.txt
#python evaluate.py 20ng energydist >> ../../../evaluations/20ng-energydist.txt
#python evaluate.py 20ng hausdorffdist >> ../../../evaluations/20ng-hausdorffdist.txt
#python evaluate.py amazon wmddist >> ../../../evaluations/amazon-wmddist.txt
#python evaluate.py amazon energydist >> ../../../evaluations/amazon-energydist.txt
#python evaluate.py amazon hausdorffdist >> ../../../evaluations/amazon-hausdorffdist.txt
#python evaluate.py classic wmddist >> ../../../evaluations/classic-wmddist.txt
#python evaluate.py classic energydist >> ../../../evaluations/classic-energydist.txt
#python evaluate.py classic hausdorffdist >> ../../../evaluations/classic-hausdorffdist.txt
#python evaluate.py ohsumed wmddist >> ../../../evaluations/ohsumed-wmddist.txt
#python evaluate.py ohsumed energydist >> ../../../evaluations/ohsumed-energydist.txt
#python evaluate.py ohsumed hausdorffdist >> ../../../evaluations/ohsumed-hausdorffdist.txt

python evaluate_baselines.py amazon eigensent >> ../../../evaluations/amazon-eigensent.txt
python evaluate_baselines.py bbcsport eigensent >> ../../../evaluations/bbcsport-eigensent.txt
python evaluate_baselines.py classic eigensent >> ../../../evaluations/classic-eigensent.txt
python evaluate_baselines.py ohsumed eigensent >> ../../../evaluations/ohsumed-eigensent.txt
python evaluate_baselines.py r8 eigensent >> ../../../evaluations/r8-eigensent.txt
python evaluate_baselines.py sst5 eigensent >> ../../../evaluations/sst5-eigensent.txt
python evaluate_baselines.py twitter eigensent >> ../../../evaluations/twitter-eigensent.txt
#python evaluate_baselines.py 20ng eigensent >> ../../../evaluations/20ng-eigensent.txt

python evaluate_baselines.py amazon dct >> ../../../evaluations/amazon-dct.txt
python evaluate_baselines.py bbcsport dct >> ../../../evaluations/bbcsport-dct.txt
python evaluate_baselines.py classic dct >> ../../../evaluations/classic-dct.txt
python evaluate_baselines.py ohsumed dct >> ../../../evaluations/ohsumed-dct.txt
python evaluate_baselines.py r8 dct >> ../../../evaluations/r8-dct.txt
python evaluate_baselines.py sst5 dct >> ../../../evaluations/sst5-dct.txt
python evaluate_baselines.py twitter dct >> ../../../evaluations/twitter-dct.txt
#python evaluate_baselines.py 20ng dct >> ../../../evaluations/20ng-dct.txt

python evaluate_baselines.py amazon powermeans >> ../../../evaluations/amazon-powermeans.txt
python evaluate_baselines.py bbcsport powermeans >> ../../../evaluations/bbcsport-powermeans.txt
python evaluate_baselines.py classic powermeans >> ../../../evaluations/classic-powermeans.txt
python evaluate_baselines.py ohsumed powermeans >> ../../../evaluations/ohsumed-powermeans.txt
python evaluate_baselines.py r8 powermeans >> ../../../evaluations/r8-powermeans.txt
python evaluate_baselines.py sst5 powermeans >> ../../../evaluations/sst5-powermeans.txt
python evaluate_baselines.py twitter powermeans >> ../../../evaluations/twitter-powermeans.txt
#python evaluate_baselines.py 20ng powermeans >> ../../../evaluations/20ng-powermeans.txt