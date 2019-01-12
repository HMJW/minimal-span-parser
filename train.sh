#!/bin/bash


datap=/data/wjiang/UCCA/ucca-to-constituency/
exe=/data/wjiang/UCCA/minimal-span-parser/src/main.py 

eval_dir=/data/wjiang/UCCA/minimal-span-parser/EVALB
train_file=$datap/English-train.txt
dev_file=$datap/English-dev.txt
log_file=./log.train.chart
save_file=./save/chart
type=chart

nohup python -u $exe train --batch-size 10 --parser-type $type --checks-per-epoch 1 --lstm-dim 150 --tag-embedding-dim 100 --label-hidden-dim 200 --split-hidden-dim 200 --model-path-base $save_file --evalb-dir $eval_dir --train-path $train_file --dev-path $dev_file  > $log_file 2>&1  &
