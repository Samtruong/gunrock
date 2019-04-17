#!/bin/bash

path="/data-2/mstruong/Frog_dataset/"
addr0=${path}"youtube.mtx"
addr1=${path}"roadNetCA.mtx"
addr2=${path}"wikiTalk.mtx"
addr3=${path}"amazon.mtx"
addr4=${path}"dblp.mtx"
addr5=${path}"twitter.mtx"
ADDR_ARRAY=($addr0 $addr1 $addr2 $addr3 $addr4 $addr5)
echo "[0] Run youtube dataset"
echo "[1] Run roadNetCA dataset"
echo "[2] Run wikiTalk dataset"
echo "[3] Run amazon dataset"
echo "[4] Run dblp dataset"
echo "[5] Run twitter dataset"
read -p "Option: " set

echo "Coloring on input frontier? (default=false color output frontier)"
read -p "Option: " inputfrontier
./bin/test_sssp_10.0_x86_64 \
--graph-type=market \
--graph-file=${ADDR_ARRAY[$set]} \
--src=largestdegree \
--device=0 \
--traversal-mode=LB \
--quick \
--color-in=$inputfrontier
#--num-run=10
