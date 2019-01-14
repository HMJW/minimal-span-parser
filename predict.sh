#!/bin/bash


model_path=./save/chart_dev=80.60
test_path=/data/wjiang/UCCA/test-data/test-xml/UCCA_English-20K
out_path=/data/wjiang/UCCA/test-data/predict-xml/UCCA_English-20K

nohup python -u src/predict.py predict --model-path-base $model_path --test-path $test_path --out-path $out_path > log.test.txt 2>&1 &