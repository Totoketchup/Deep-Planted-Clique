#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc nvcr-tensorflow-1707_g1 
#$ -e err_rnn.o
#$ -o std_rnn.o

. /fefs/opt/dgx/env_set/common_env_set.sh

python rnn_planted.py --data clique-N1000-K25-E10-M1-exFalse-L:False-F:False