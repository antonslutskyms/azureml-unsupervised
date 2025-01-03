#!/bin/bash

#conda install -c rapidsai -c nvidia -c conda-forge cuml=23.02 python=3.8 cudatoolkit=11.2
#conda install pandas



echo "hello"

echo $*

CLE="kmeans_clustering-copy.py"

echo $CLE

base_folder=$1

echo $base_folder

out_folder=$2
echo $out_folder


final=$3

echo $final

echo $*

python -u $CLE -curated_dataset $base_folder      \
               -odir $out_folder      \
               -final $final