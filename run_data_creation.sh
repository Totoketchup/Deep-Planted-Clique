#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc nvcr-tensorflow-1707_g1 
#$ -e err_dat.o
#$ -o std_dat.o

. /fefs/opt/dgx/env_set/common_env_set.sh

./generate_data