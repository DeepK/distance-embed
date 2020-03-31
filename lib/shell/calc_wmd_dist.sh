cd ../py/distances

python2 prepare_data_for_wmd.py bbcsport
python2 wmd/wmd.py ../produced/bbcsport_vec.pk ../produced/wmddist_bbcsport.pk

python2 prepare_data_for_wmd.py twitter
python2 wmd/wmd.py ../produced/twitter_vec.pk ../produced/wmddist_twitter.pk

python2 prepare_data_for_wmd.py r8
python2 wmd/wmd.py ../produced/r8_vec.pk ../produced/wmddist_r8.pk

python2 prepare_data_for_wmd.py sst5
python2 wmd/wmd.py ../produced/sst5_vec.pk ../produced/wmddist_sst5.pk

python2 prepare_data_for_wmd.py amazon
python2 wmd/wmd.py ../produced/amazon_vec.pk ../produced/wmddist_amazon.pk

python2 prepare_data_for_wmd.py classic
python2 wmd/wmd.py ../produced/classic_vec.pk ../produced/wmddist_classic.pk

python2 prepare_data_for_wmd.py ohsumed
python2 wmd/wmd.py ../produced/ohsumed_vec.pk ../produced/wmddist_ohsumed.pk
