alias python=python3

cd ../py/embeddings
python umap_embed.py bbcsport energydist
python umap_embed.py bbcsport wmddist
python umap_embed.py bbcsport hausdorffdist

python umap_embed.py twitter wmddist
python umap_embed.py twitter energydist
python umap_embed.py twitter hausdorffdist

python umap_embed.py r8 wmddist
python umap_embed.py r8 energydist
python umap_embed.py r8 hausdorffdist

python umap_embed.py sst5 wmddist
python umap_embed.py sst5 energydist
python umap_embed.py sst5 hausdorffdist

python umap_embed.py amazon wmddist
python umap_embed.py amazon energydist
python umap_embed.py amazon hausdorffdist

python umap_embed.py classic wmddist
python umap_embed.py classic energydist
python umap_embed.py classic hausdorffdist

python umap_embed.py ohsumed wmddist
python umap_embed.py ohsumed energydist
python umap_embed.py ohsumed hausdorffdist
