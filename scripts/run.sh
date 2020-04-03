#!/bin/sh
source /etc/profile
source /etc/bashrc
source ~/.bash_profile
cd /data/services/recsys_learning
pyenv shell 3.6.8
python modules/ncf/DataPrepare.py
python modules/ncf/NCF.py --path=/data/services/recsys_learning/datas/info --dataset=info --epochs=16 --num_factors=8
ossutil cp -rf emb.txt oss://kobo-recsys/di/ncf/emb.txt
