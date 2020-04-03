#!/bin/sh
source /etc/profile
source /etc/bashrc
source ~/.bash_profile
cd /data/services/recsys_learning
pyenv shell 3.6.8
python modules/ncf/DataPrepare.py
python modules/ncf/NCF.py --path=/data/services/recsys_learning/datas/info --dataset=info --epochs=16 --num_factors=8 --layers=[32,16,8]
python modules/ncf/sim.user.py 8 datas/info/info.uid.map.txt emb.txt sim.user.txt
ossutil cp -rf sim.user.txt oss://kobo-recsys/di/ncf/sim.user.txt
