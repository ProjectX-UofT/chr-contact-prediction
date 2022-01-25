#!/bin/sh

mkdir -p raw

for i in $(seq 0 27)
do
    echo $i
    wget -O raw/train-$i.tfr "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/train-$i.tfr"
done
for i in 0 1
do
    wget -O raw/valid-$i.tfr "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/valid-$i.tfr"
done
for i in 0 1
do
    wget -O raw/test-$i.tfr "https://storage.googleapis.com/basenji_hic/1m/data/tfrecords/test-$i.tfr"
done
