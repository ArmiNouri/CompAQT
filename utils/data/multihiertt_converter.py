import os
import json


if __name__ == "__main__":

    path = '../../datasets/multihiertt/train.json'
    samples = []
    with open(path, 'r') as o:
        j = json.load(o)
        for sample in j:
            if sample['qa']['question_type'] != 'arithmetic': continue
            samples.append(sample['qa']['program'])
    print('train', len(samples))

    path = '../../datasets/multihiertt/val.json'
    samples = []
    with open(path, 'r') as o:
        j = json.load(o)
        for sample in j:
            if sample['qa']['question_type'] != 'arithmetic': continue
            samples.append(sample['qa']['program'])
    print('dev', len(samples))