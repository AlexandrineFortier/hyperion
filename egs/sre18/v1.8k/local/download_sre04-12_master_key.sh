#!/bin/bash

# Copyright 2019 Johns Hopkins University (Jesus Villalba)
# Apache 2.0
#
# Downloads master key for sre04-12.
set -e 
key_name=master_key_sre04-12
master_key=$key_name/NIST_SRE_segments_key.csv

# shareable link:
# https://drive.google.com/file/d/1ukJNhj0-7_uWsIFWvtEFQHweq-lFG5a7/view?usp=sharing

wget --no-check-certificate "https://drive.google.com/uc?export=download&id=1ukJNhj0-7_uWsIFWvtEFQHweq-lFG5a7" -O $key_name.tar.gz
tar xzvf $key_name.tar.gz

if [ ! -f $master_key ];then
    echo "master key wasn't dowloaded correctly"
    exit 1
fi

rm -f $key_name.tar.gz
