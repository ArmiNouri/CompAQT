# coding=utf-8
import os
import sys
import random
import argparse
import json


my_parser = argparse.ArgumentParser(description="Merge all four datasets into one.")
my_parser.add_argument("--finqa", "-f", dest="finqa_path", type=str, help="The path to the FinQA directory.")
my_parser.add_argument("--tatqa", "-t", dest="tatqa_path", type=str, help="The path to the TAT-QA directory.")
my_parser.add_argument("--hitab", "-b", dest="hitab_path", type=str, help="The path to the HiTab directory.")
my_parser.add_argument("--multihiertt", "-m", dest="multi_path", type=str, help="The path to the MultiHiertt directory.")
my_parser.add_argument("--output", "-o", dest="output_path", type=str, help="The path to output directory.")


if __name__ == "__main__":
    args = my_parser.parse_args()
    # Load FinQA
    with open(os.path.join(args.finqa_path, 'train_retrieve.json'), 'r') as f:
        finqa_train = json.load(f)
    with open(os.path.join(args.finqa_path, 'dev_retrieve.json'), 'r') as f:
        finqa_dev = json.load(f)
    with open(os.path.join(args.finqa_path, 'test_retrieve.json'), 'r') as f:
        finqa_test = json.load(f)
    # Load TAT-QA
    with open(os.path.join(args.tatqa_path, 'tatqa_dataset_train_finqa.json'), 'r') as f:
        tatqa_train = json.load(f)
    # TAT-QA dev set to be split 230-307
    with open(os.path.join(args.tatqa_path, 'tatqa_dataset_dev_finqa.json'), 'r') as f:
        tatqa = json.load(f)
        random.shuffle(tatqa)
        tatqa_dev = tatqa[:230]
        tatqa_test = tatqa[230:]
    # Load HiTab
    with open(os.path.join(args.hitab_path, 'hitab_dataset_train_finqa.json'), 'r') as f:
        hitab_train = json.load(f)
    with open(os.path.join(args.hitab_path, 'hitab_dataset_dev_finqa.json'), 'r') as f:
        hitab_dev = json.load(f)
    with open(os.path.join(args.hitab_path, 'hitab_dataset_test_finqa.json'), 'r') as f:
        hitab_test = json.load(f)
    # Load MultiHiertt
    with open(os.path.join(args.multi_path, 'multihiertt_dataset_train_finqa.json'), 'r') as f:
        multi_train = json.load(f)
    # MultiHiertt dev set to be split 200-337
    with open(os.path.join(args.multi_path, 'multihiertt_dataset_dev_finqa.json'), 'r') as f:
        multi = json.load(f)
        random.shuffle(multi)
        multi_dev = multi[:100]
        multi_test = multi[100:]

    print('FinQA:', len(finqa_train), len(finqa_dev), len(finqa_test))
    print('TAT-QA:', len(tatqa_train), len(tatqa_dev), len(tatqa_test))
    print('HiTab:', len(hitab_train), len(hitab_dev), len(hitab_test))
    print('MultiHiertt:', len(multi_train), len(multi_dev), len(multi_test))

    for x in finqa_train: x['source'] = "finqa"
    for x in finqa_dev: x['source'] = "finqa"
    for x in finqa_test: x['source'] = "finqa"

    for x in tatqa_train: x['source'] = "tatqa"
    for x in tatqa_dev: x['source'] = "tatqa"
    for x in tatqa_test: x['source'] = "tatqa"

    for x in hitab_train: x['source'] = "hitab"
    for x in hitab_dev: x['source'] = "hitab"
    for x in hitab_test: x['source'] = "hitab"

    for x in multi_train: x['source'] = "multihiertt"
    for x in multi_dev: x['source'] = "multihiertt"
    for x in multi_test: x['source'] = "multihiertt"

    train = finqa_train + tatqa_train + hitab_train + multi_train
    dev = finqa_dev + tatqa_dev + hitab_dev + multi_dev
    test = finqa_test + tatqa_test + hitab_test + multi_test

    with open(os.path.join(args.output_path, 'train.json'), 'w') as f:
        f.write(json.dumps(train, indent=4))

    with open(os.path.join(args.output_path, 'dev.json'), 'w') as f:
        f.write(json.dumps(dev, indent=4))

    with open(os.path.join(args.output_path, 'test.json'), 'w') as f:
        f.write(json.dumps(test, indent=4))
    
    print('Done!')
