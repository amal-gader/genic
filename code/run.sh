#!/bin/bash

MODE=$1 
DATASET=$2
ID_LP=$3
ID_PP=$4
WITH_TYPE=$5
WITH_DESC=$6

if [ $MODE = "train" ]
then
    echo "Start Training......"

    python main.py --mode "$MODE" --dataset "$DATASET" \
        $(if $WITH_TYPE; then echo "--with_type"; fi) \
        $(if $WITH_DESC; then echo "--with_desc"; fi)

elif [ $MODE = "test" ]
then
    echo "Start Testing......"

    python main.py --mode "$MODE" --dataset "$DATASET" \
        $(if [ -n "$ID_LP" ]; then echo "--id_lp $ID_LP"; fi) \
        $(if [ -n "$ID_PP" ]; then echo "--id_pp $ID_PP"; fi) \
        $(if $WITH_TYPE; then echo "--with_type"; fi) \
        $(if $WITH_DESC; then echo "--with_desc"; fi)

else
    echo "Unknown MODE: $MODE"
fi
