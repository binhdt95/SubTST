#!/bin/bash


MODEL='bert-base-cased'
DATASET='stsB'
NUM_TOPIC=100
#Scripts of SubTST
python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling mean --use-topic true --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling max --use-topic true --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling mean --use-topic true --train-topic true --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling max --use-topic true --train-topic true --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
wait

