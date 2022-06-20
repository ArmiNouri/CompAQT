import os
import json


if __name__ == "__main__":
    table_ids = set()

    path = '../../datasets/hitab/train_samples.jsonl'
    samples = []
    with open(path, 'r') as o:
        for line in o.readlines():
            j = json.loads(line)
            f = j['answer_formulas']
            if len(f) > 1: continue
            f = str(f[0])
            if '(' not in f and ')' not in f and '+' not in f and '-' not in f and '/' not in f and '*' not in f: continue
            samples.append(f)
            table_ids.add(j['table_id'])
    print('train', len(samples))

    path = '../../datasets/hitab/dev_samples.jsonl'
    samples = []
    with open(path, 'r') as o:
        for line in o.readlines():
            j = json.loads(line)
            f = j['answer_formulas']
            if len(f) > 1: continue
            f = str(f[0])
            if '(' not in f and ')' not in f and '+' not in f and '-' not in f and '/' not in f and '*' not in f: continue
            samples.append(f)
            table_ids.add(j['table_id'])
    print('dev', len(samples))

    path = '../../datasets/hitab/test_samples.jsonl'
    samples = []
    with open(path, 'r') as o:
        for line in o.readlines():
            j = json.loads(line)
            f = j['answer_formulas']
            if len(f) > 1: continue
            f = str(f[0])
            if '(' not in f and ')' not in f and '+' not in f and '-' not in f and '/' not in f and '*' not in f: continue
            samples.append(f)
            table_ids.add(j['table_id'])
    print('test', len(samples))

    print(len(table_ids))
    