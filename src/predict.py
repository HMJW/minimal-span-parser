import argparse
import itertools
import os
import os.path
import time
from collections import Counter

import numpy as np

import dynet as dy
import evaluate
import parse
import trees
import vocabulary
from ucca.convert import to_text, xml2passage
from ucca.core import edge_id_orderkey
from ucca.layer1 import FoundationalNode


def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string


def read_passages(path):
    passages = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            print(file_path)
        passages.append(xml2passage(file_path))
    return passages


def read_raw(path):
    raw = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            print(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            sen = f.readline()
            raw.append(sen.split())
    return raw


def get_input(passage):
    terminals = sorted(
        [node for node in passage.layer("0").all], key=lambda x: x.position
    )
    return [(t.extra["pos"], t.text) for t in terminals]


def run_predict(args):
    print("Loading test xmls from {}...".format(args.test_path))
    raws = read_raw('../test-data/test-txt/UCCA_English-20K/')
    passages = read_passages(args.test_path)
    print("Loaded {:,} test examples.".format(len(passages)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()

    test_predicted = []
    for passage in passages:
        dy.renew_cg()
        sentence = get_input(passage)
        predicted, _ = parser.parse(sentence)
        test_predicted.append(predicted.convert())
    return test_predicted

def predict():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("predict")
    subparser.set_defaults(callback=run_predict)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument(
        "--test-path", default="/data/wjiang/UCCA/test-data/test-xml/UCCA_English-20K"
    )

    args = parser.parse_args()
    predicted = args.callback(args)
    return predicted


if __name__ == "__main__":
    predicted = predict()
    print(predicted[0].linearize())