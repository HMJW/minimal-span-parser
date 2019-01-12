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
from ucca.convert import to_text, xml2passage, passage2file
from ucca.core import edge_id_orderkey
from ucca.layer1 import Layer1, PunctNode, NodeTags
from ucca.layer0 import Terminal


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
    return passages, test_predicted


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
    passages, predicted = args.callback(args)
    return passages, predicted


def tree2passage(passage, tree):
    def add_node(passage, node, tree):
        global leaf_position
        if isinstance(tree, trees.LeafTreebankNode):
            return
        for c in tree.children:
            if isinstance(c, trees.LeafTreebankNode):
                node.add("Terminal", passage.layer("0").by_position(leaf_position))
                leaf_position += 1
            else:
                new_fnode = passage.layer("1").add_fnode(node, c.label)
                add_node(passage, new_fnode, c)

    def change_puncnode(passage):
        for t in passage.layer("0")._all:
            parent = t.parents[0]
            children = parent.children
            is_punc = sum(isinstance(c, Terminal) and c.punct for c in children)
            if is_punc == len(children):
                true_ID = parent.ID
                punc_node = PunctNode(
                    root=passage,
                    tag=NodeTags.Punctuation,
                    ID=passage.layer("1").next_id(),
                )
                parent.parents[0].add(parent._incoming[0].tag, punc_node)
                for c in children:
                    punc_node.add("Terminal", c)

                for i in parent._incoming:
                    parent.parents[0]._outgoing.remove(i)
                for i in parent._outgoing:
                    i.child._incoming.remove(i)
                passage.layer("1")._all.remove(parent)
                punc_node._ID = true_ID

    if "format" not in passage.extra:
        passage.extra["format"] = "ucca"
    layer = Layer1(passage, attrib=None)
    assert "0" in passage._layers
    if tree.label != "ROOT":
        print("warning: root label is not ROOT but %s" % tree.label)

    global leaf_position
    leaf_position = 1
    add_node(passage, layer._head_fnode, tree)
    change_puncnode(passage)


def restore_discontinuity(passage):
    def restore_down(node, e):
        if len(node.parents) == 1 and len(node.parents[0].parents) == 1:
            # assert len(node.parents) == 1
            node.parents[0]._outgoing.remove(e)
            node.parents[0].parents[0]._outgoing.append(e)
            e._parent = node.parents[0].parents[0]
            node._outgoing.sort(key=edge_id_orderkey)
        e._tag = e._tag.strip("-down")

    def restore_left(node, e):
        assert len(node.parents) == 1
        parent = node.parents[0]
        children = list(
            sorted(parent.children, key=lambda x: x.get_terminals()[0].position)
        )
        from_index = children.index(node)
        if len(children) > 1:
            to_node = children[from_index - 1]
            node.parents[0]._outgoing.remove(e)
            e._parent = to_node
            to_node._outgoing.append(e)
            to_node._outgoing.sort(key=edge_id_orderkey)
        e._tag = e._tag.strip("-left")

    for node in passage.layer("1")._all:
        for i in node._incoming:
            if "down" in i.tag:
                print(passage.ID, i._tag)
                restore_down(node, i)
            elif "left" in i.tag:
                restore_left(node, i)


def restore_remote(passage):
    pass


def to_UCCA(passage, tree):
    tree2passage(passage, tree)
    restore_discontinuity(passage)
    restore_remote(passage)


if __name__ == "__main__":
    passages, predicted = predict()
    for x, y in zip(passages, predicted):
        to_UCCA(x, y)
    print(passages[0])
    print(predicted[0].linearize())
    passage2file(passages[0], "./" + passages[0].ID + ".xml")

