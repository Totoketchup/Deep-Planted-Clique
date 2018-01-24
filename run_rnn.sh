#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc nvcr-tensorflow-1707_g1 
#$ -e err_rnn.o
#$ -o std_rnn.o

export MY_PROXY_URL="http://10.1.10.1:8080/"
export HTTP_PROXY=$MY_PROXY_URL
export HTTPS_PROXY=$MY_PROXY_URL
export FTP_PROXY=$MY_PROXY_URL
export http_proxy=$MY_PROXY_URL 
export https_proxy=$MY_PROXY_URL
export ftp_proxy=$MY_PROXY_URL

. /fefs/opt/dgx/env_set/common_env_set.sh

apt-get source python-tk
./configure
make 
make install

pip install --user -r requirements.txt
python rnn_planted.py --data clique-N1000-K25-E10-M1-exFalse-L:False-F:False