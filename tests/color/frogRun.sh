#!/bin/bash

path="/data-2/mstruong/Frog_dataset/"
addr0=${path}"youtube.mtx"
addr1=${path}"roadNetCA.mtx"
addr2=${path}"wikiTalk.mtx"
addr3=${path}"amazon.mtx"
addr4=${path}"dblp.mtx"
addr5=${path}"twitter.mtx"
ADDR_ARRAY=($addr0 $addr1 $addr2 $addr3 $addr4)


	echo "[0] Run youtube dataset"
	echo "[1] Run roadNetCA dataset"
	echo "[2] Run wikiTalk dataset"
	echo "[3] Run amazon dataset"
	echo "[4] Run dblp dataset"
	echo "[5] Run twitter dataset"
	read -p "Option: " set
	./bin/test_color_10.0_x86_64 \
	--graph-type=market \
	--graph-file=${ADDR_ARRAY[$set]} \
	--JPL=true \
	--no-conflict=0 \
	--user-iter=0 \
	--prohibit-size=0 \
	--quick=true \
	--device=0 \
	--min-color=true \
	--test-run=true \
	--undirected \
	--remove-duplicate-edges \
	--remove-self-loops \
        --loop-color=true \
        --check-percentage=true
