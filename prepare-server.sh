sudo apt-get install python-minimal
sudo apt-get install python3
sudo apt update
sudo apt upgrade
sudo apt install python2.7 python-pip
sudo apt install python3-pip
sudo apt-get install swig

cd lib
pip2 install -r requirements.txt
pip3 install -r requirements.txt
python3 setup.py install

cd ..
mkdir evaluation-knn
mkdir evaluations
mkdir exact_embeddings
mkdir produced
mkdir resources

cd resources
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"

cd ..
cd lib/py/distances/emd/
make