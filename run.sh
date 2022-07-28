#!/bin/bash
export PATH=/home/binhdt/s1920422/anaconda3/bin:$PATH
# export PATH=/home/s1920422/anaconda3/bin:$PATH

export CUDA_VISIBLE_DEVICES=0
python test_msrp.py 
# python test_subtaskA.py


#Scripts of SBERT
# python sts.py --model bert-base-cased --dataset SICK --path-data data/SICK --pooling mean --use-topic false --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 &
# python sts.py --model bert-base-cased --dataset SICK --path-data data/SICK --pooling max --use-topic false --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 &


# MODEL='bert-base-cased'
# DATASET='stsB'
# NUM_TOPIC=250
# #Scripts of SubTST
# python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling mean --use-topic true --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
# python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling max --use-topic true --train-topic false --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
# # python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling mean --use-topic true --train-topic true --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
# # python sts.py --model $MODEL --dataset $DATASET --path-data data/$DATASET --pooling max --use-topic true --train-topic true --train-file train.csv --dev-file dev.csv --test-file test.csv --batch-size 16 --epochs 3 --num-topics $NUM_TOPIC &
# wait

