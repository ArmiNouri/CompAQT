"""MathQA utils.
"""
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip

from config import parameters as conf


# verbalization of operators
# add, subtract, multiply, divide, exp, greater, table_sum, table_average, table_max, table_min
verbalization_map = {'add': 'add', 'subtract': 'subtract', 'multiply': 'multiply', 'divide': 'divide', 
                    'exp': 'exponentiate', 'greater': 'larger', 
                    'table_sum': 'table sum', 'table_average': 'table average', 'table_max': 'table max', 'table_min': 'table min',
                    'UNK': 'unknown', 'GO': 'start', 'EOF': 'done', ')': 'close'}


def str_to_num(text):
    text = text.replace(",", "")
    try:
        num = int(text)
    except ValueError:
        try:
            num = float(text)
        except ValueError:
            if text and text[-1] == "%":
                num = text
            else:
                num = None
    return num


def prog_token_to_indices(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(op_list_size + const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            # print(prog)
            # print(token)
            # print(const_list)
            # print(numbers)
            assert cur_num_idx != -1
            prog_indices.append(op_list_size + const_list_size +
                                number_indices[cur_num_idx])
    return prog_indices


def prog_token_to_indices_simple(prog, numbers, number_indices, max_seq_length,
                          op_list, op_list_size, const_list,
                          const_list_size):
    prog_indices = []
    for i, token in enumerate(prog):
        if token in op_list:
            prog_indices.append(op_list.index(token))
        elif token in const_list:
            prog_indices.append(const_list.index(token))
        else:
            if token in numbers:
                cur_num_idx = numbers.index(token)
            else:
                cur_num_idx = -1
                for num_idx, num in enumerate(numbers):
                    if str_to_num(num) == str_to_num(token):
                        cur_num_idx = num_idx
                        break
            # print(prog)
            # print(token)
            # print(const_list)
            # print(numbers)
            assert cur_num_idx != -1
            prog_indices.append(const_list_size + # 1 +  # account for [CLS] token
                                number_indices[cur_num_idx])
    return prog_indices


def indices_to_prog(program_indices, numbers, number_indices, max_seq_length,
                    op_list, op_list_size, const_list, const_list_size):
    prog = []
    for i, prog_id in enumerate(program_indices):
        if prog_id < op_list_size:
            prog.append(op_list[prog_id])
        elif prog_id < op_list_size + const_list_size:
            prog.append(const_list[prog_id - op_list_size])
        else:
            prog.append(numbers[number_indices.index(prog_id - op_list_size
                                                     - const_list_size)])
    return prog


class MathQAExample(
        collections.namedtuple(
            "MathQAExample",
            "source id original_question question_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example(self, *args, **kwargs)


class MathQAExampleSimple(
        collections.namedtuple(
            "MathQAExample",
            "source id question question_tokens facts facts_tokens options answer \
            numbers number_indices original_program program"
        )):

    def convert_single_example(self, *args, **kwargs):
        return convert_single_mathqa_example_simple(self, *args, **kwargs)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 question,
                 input_ids,
                 input_mask,
                 option_mask,
                 segment_ids,
                 options,
                 answer=None,
                 program=None,
                 program_ids=None,
                 program_weight=None,
                 program_mask=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.question = question
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.option_mask = option_mask
        self.segment_ids = segment_ids
        self.options = options
        self.answer = answer
        self.program = program
        self.program_ids = program_ids
        self.program_weight = program_weight
        self.program_mask = program_mask


class InputFeaturesSimple(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 question_tokens,
                 question_ids,
                 question_mask,
                 facts_tokens,
                 facts_ids,
                 facts_mask,
                 numbers_mask,
                 answer=None,
                 program=None,
                 program_ids=None,
                 program_weight=None,
                 program_mask=None,
                 operator_ids=None,
                 operator_mask=None,
                 operator_list_ids=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.question_tokens = question_tokens
        self.question_ids = question_ids
        self.question_mask = question_mask
        self.facts_tokens = facts_tokens
        self.facts_ids = facts_ids
        self.facts_mask = facts_mask
        self.numbers_mask = numbers_mask
        self.answer = answer
        self.program = program
        self.program_ids = program_ids
        self.program_weight = program_weight
        self.program_mask = program_mask
        self.operator_ids = operator_ids
        self.operator_mask = operator_mask


def tokenize(tokenizer, text, apply_basic_tokenization=False):
    """Tokenizes text, optionally looking up special tokens separately.

    Args:
      tokenizer: a tokenizer from bert.tokenization.FullTokenizer
      text: text to tokenize
      apply_basic_tokenization: If True, apply the basic tokenization. If False,
        apply the full tokenization (basic + wordpiece).

    Returns:
      tokenized text.

    A special token is any text with no spaces enclosed in square brackets with no
    space, so we separate those out and look them up in the dictionary before
    doing actual tokenization.
    """

    if conf.pretrained_model in ["bert", "finbert"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)
    elif conf.pretrained_model in ["roberta", "longformer"]:
        _SPECIAL_TOKENS_RE = re.compile(r"^<[^ ]*>$", re.UNICODE)

    tokenize_fn = tokenizer.tokenize
    if apply_basic_tokenization:
        tokenize_fn = tokenizer.basic_tokenizer.tokenize

    tokens = []
    for token in text.split(" "):
        if _SPECIAL_TOKENS_RE.match(token):
            if token in tokenizer.get_vocab():
                tokens.append(token)
            else:
                tokens.append(tokenizer.unk_token)
        else:
            tokens.extend(tokenize_fn(token))

    return tokens


def _detokenize(tokens):
    text = " ".join(tokens)

    text = text.replace(" ##", "")
    text = text.replace("##", "")

    text = text.strip()
    text = " ".join(text.split())
    return text


def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                    cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program


def convert_single_mathqa_example(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    if len(question_tokens) > max_seq_length - 2:
        print("too long")
        question_tokens = question_tokens[:max_seq_length - 2]
    tokens = [cls_token] + question_tokens + [sep_token]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < len(input_mask):
            input_mask[offset] = 2
        else:
            if is_training == True:

                # invalid example, drop for training
                return features

            # assert is_training == False

    padding = [0] * (max_seq_length - len(input_ids))
    input_ids.extend(padding)
    input_mask.extend(padding)
    segment_ids.extend(padding)

    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    number_mask = [tmp - 1 for tmp in input_mask]
    for ind in range(len(number_mask)):
        if number_mask[ind] < 0:
            number_mask[ind] = 0
    option_mask = [1, 0, 0, 1] + [1] * (len(op_list) + len(const_list) - 4)
    option_mask = option_mask + number_mask
    option_mask = [float(tmp) for tmp in option_mask]

    for ind in range(len(input_mask)):
        if input_mask[ind] > 1:
            input_mask[ind] = 1

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        try:
            program_ids = prog_token_to_indices(program, numbers, number_indices,
                                                max_seq_length, op_list, op_list_size,
                                                const_list, const_list_size)
            program_mask = [1] * len(program_ids)
            program_ids = program_ids[:max_program_length]
            program_mask = program_mask[:max_program_length]
            if len(program_ids) < max_program_length:
                padding = [0] * (max_program_length - len(program_ids))
                program_ids.extend(padding)
                program_mask.extend(padding)
        except AssertionError:
            print("NUMBER NOT FOUND!", example.id, program, numbers)
            program = ""
            program_ids = [0] * max_program_length
            program_mask = [0] * max_program_length
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length
    features.append(
        InputFeatures(
            unique_id=-1,
            example_index=-1,
            tokens=tokens,
            question=example.original_question,
            input_ids=input_ids,
            input_mask=input_mask,
            option_mask=option_mask,
            segment_ids=segment_ids,
            options=example.options,
            answer=example.answer,
            program=program,
            program_ids=program_ids,
            program_weight=1.0,
            program_mask=program_mask))
    return features


def convert_single_mathqa_example_simple(example, is_training, tokenizer, max_seq_length,
                                  max_program_length, op_list, op_list_size,
                                  const_list, const_list_size,
                                  cls_token, sep_token):
    """Converts a single MathQAExample into an InputFeature."""
    features = []
    question_tokens = example.question_tokens
    # leave room for CLS
    if len(question_tokens) > max_seq_length - 1:
        print("question too long; truncating....")
        question_tokens = question_tokens[:max_seq_length - 1]
    question_tokens = [cls_token] + question_tokens
    ql = len(question_tokens)
    question_tokens = question_tokens + [tokenizer.pad_token]*(max_seq_length-ql) # padding
    question_ids = tokenizer.convert_tokens_to_ids(question_tokens)
    question_mask = [False] * ql + [True] * (max_seq_length-ql) # padding mask

    operator_indices = [x*4 for x in range(max_program_length//4)]
    operator_programs = [example.program[i] for i in operator_indices if i < len(example.program)]
    operator_programs = [x.replace('(', '') for x in operator_programs]     # remove '('
    operator_tokens = [verbalization_map[x] for x in operator_programs]
    operator_tokens = tokenizer.tokenize(' '.join(operator_tokens)) # max length: 7 * 3 = 21
    operator_tokens = [tokenizer.cls_token] + operator_tokens  # 1 + 21 = 22

    max_op_length = max_program_length//4
    max_op_length = 1 + max_op_length * 3
    if len(operator_tokens) > max_op_length:
        print("operators too long; truncating....")
        operator_tokens = operator_tokens[:max_op_length]
    ol = len(operator_tokens)
    operator_tokens = operator_tokens + [tokenizer.pad_token]*(max_op_length-ol)
    # operator_ids = tokenizer.convert_tokens_to_ids(operator_tokens)
    # operator_mask = [False] * ol + [True] * (max_op_length-ol) # padding mask
    
    # for k, v in verbalization_map.items(): print(k, tokenizer.convert_tokens_to_ids([v]))

    facts_tokens = example.facts_tokens
    # leave room for CLS
    if len(facts_tokens) > max_seq_length - 1:
        print("facts too long; truncating....")
        facts_tokens = facts_tokens[:max_seq_length - 1]
    facts_tokens = [cls_token] + facts_tokens
    fl = len(facts_tokens)
    facts_tokens = facts_tokens + [tokenizer.pad_token]*(max_seq_length-fl) # padding
    facts_ids = tokenizer.convert_tokens_to_ids(facts_tokens)
    facts_mask = [False]*fl + [True]*(max_seq_length-fl) # padding mask

    numbers_mask = [False] * len(facts_ids)
    for ind, offset in enumerate(example.number_indices):
        if offset < fl:
            numbers_mask[offset] = True
        else:
            if is_training == True:

                # invalid example, drop for training
                print("Invalid example!")
                return features

    assert len(facts_ids) == max_seq_length
    assert len(facts_mask) == max_seq_length

    numbers = example.numbers
    number_indices = example.number_indices
    program = example.program
    if program is not None and is_training:
        try:
            program_ids = prog_token_to_indices_simple(program, numbers, number_indices,
                                                max_seq_length, op_list, op_list_size,
                                                const_list, const_list_size)
            # print(facts_tokens)
            # print(program)
            # print(program_ids)
            # print('=================')
            program_mask = [1] * len(program_ids)
            program_ids = program_ids[:max_program_length]
            program_mask = program_mask[:max_program_length]
            if len(program_ids) < max_program_length:
                padding = [0] * (max_program_length - len(program_ids))
                program_ids.extend(padding)
                program_mask.extend(padding)
        except AssertionError:
            print("NUMBER NOT FOUND!", program, numbers)
            program = ""
            program_ids = [0] * max_program_length
            program_mask = [0] * max_program_length
    else:
        program = ""
        program_ids = [0] * max_program_length
        program_mask = [0] * max_program_length
    assert len(program_ids) == max_program_length
    assert len(program_mask) == max_program_length

    operator_ids = [program_ids[i*4] for i in range(max_program_length//4)]
    operator_mask = [program_mask[i*4] for i in range(max_program_length//4)]

    features.append(
        InputFeaturesSimple(
            unique_id=-1,
            example_index=-1,
            question_tokens=question_tokens,
            question_ids=question_ids,
            question_mask=question_mask,
            facts_tokens=facts_tokens,
            facts_ids=facts_ids,
            facts_mask=facts_mask,
            numbers_mask=numbers_mask,
            answer=example.answer,
            program=program,
            program_ids=program_ids,
            program_weight=1.0,
            program_mask=program_mask,
            operator_ids =operator_ids,
            operator_mask=operator_mask))
    return features


def remove_space(text_in):
    res = []

    for tmp in text_in.split(" "):
        if tmp != "":
            res.append(tmp)

    return " ".join(res)


def table_row_to_text(header, row):
    '''
    use templates to convert table row to text
    '''
    res = ""

    for head, cell in zip(header[1:], row[1:]):
        res += ("the " + row[0] + " of " + head + " is " + cell + " ; ")

    res = remove_space(res)
    return res.strip()


def read_mathqa_entry(entry, tokenizer):

    question = entry["qa"]["question"]
    this_id = entry["id"]
    context = ""

    if conf.retrieve_mode == "single":
        for ind, each_sent in entry["qa"]["model_input"]:
            context += each_sent
            context += " "
    elif conf.retrieve_mode == "slide":
        if len(entry["qa"]["pos_windows"]) > 0:
            context = random.choice(entry["qa"]["pos_windows"])[0]
        else:
            context = entry["qa"]["neg_windows"][0][0]
    elif conf.retrieve_mode == "gold":
        for each_con in entry["qa"]["gold_inds"]:
            context += entry["qa"]["gold_inds"][each_con]
            context += " "

    elif conf.retrieve_mode == "none":
        # no retriever, use longformer
        table = entry["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        context = " ".join(entry["pre_text"]) + " " + \
            " ".join(entry["post_text"]) + " " + table_text

    context = context.strip()
    # process "." and "*" in text
    context = context.replace(". . . . . .", "")
    context = context.replace("* * * * * *", "")

    original_question = question + " " + tokenizer.sep_token + " " + context.strip()

    options = entry["qa"]["exe_ans"]
    original_question_tokens = original_question.split(' ')

    numbers = []
    number_indices = []
    question_tokens = []
    for i, tok in enumerate(original_question_tokens):
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            number_indices.append(len(question_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(question_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        question_tokens.extend(tok_proc)

        answer = entry["qa"]["exe_ans"]

    # table headers
    for row in entry["table"]:
        tok = row[0]
        if tok and tok in original_question:
            numbers.append(tok)
            tok_index = original_question.index(tok)
            prev_tokens = original_question[:tok_index]
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)

    if conf.program_mode == "seq":
        if 'program' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)

    elif conf.program_mode == "nest":
        if 'program_re' in entry["qa"]:
            original_program = entry["qa"]['program_re']
            program = program_tokenization(original_program)

    else:
        program = None

    return MathQAExample(
        source=entry['source'],
        id=this_id,
        original_question=original_question,
        question_tokens=question_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)


def read_mathqa_entry_simple(entry, tokenizer):

    question = entry["qa"]["question"]
    this_id = entry["id"]
    facts = ""

    if conf.retrieve_mode == "single":
        for ind, each_sent in entry["qa"]["model_input"]:
            facts += each_sent
            facts += " "
    elif conf.retrieve_mode == "gold":
        for each_con in entry["qa"]["gold_inds"]:
            facts += entry["qa"]["gold_inds"][each_con]
            facts += " "

    elif conf.retrieve_mode == "none":
        # no retriever, use longformer
        table = entry["table"]
        table_text = ""
        for row in table[1:]:
            this_sent = table_row_to_text(table[0], row)
            table_text += this_sent

        facts = " ".join(entry["pre_text"]) + " " + \
            " ".join(entry["post_text"]) + " " + table_text

    facts = facts.strip()
    # process "." and "*" in text
    facts = facts.replace(". . . . . .", "")
    facts = facts.replace("* * * * * *", "")
    facts = facts.strip()

    options = entry["qa"]["exe_ans"]
    tokens = facts.split(' ')

    numbers = []
    number_indices = []
    facts_tokens = []
    for i, tok in enumerate(tokens):
        num = str_to_num(tok)
        if num is not None:
            numbers.append(tok)
            number_indices.append(len(facts_tokens))
            if tok[0] == '.':
                numbers.append(str(str_to_num(tok[1:])))
                number_indices.append(len(facts_tokens) + 1)
        tok_proc = tokenize(tokenizer, tok)
        facts_tokens.extend(tok_proc)

        answer = entry["qa"]["exe_ans"]

    # table headers
    for row in entry["table"]:
        tok = row[0]
        if tok and tok in facts:
            numbers.append(tok)
            tok_index = facts.index(tok)
            prev_tokens = facts[:tok_index]
            number_indices.append(len(tokenize(tokenizer, prev_tokens)) + 1)

    if conf.program_mode == "seq":
        if 'program' in entry["qa"]:
            original_program = entry["qa"]['program']
            program = program_tokenization(original_program)

    elif conf.program_mode == "nest":
        if 'program_re' in entry["qa"]:
            original_program = entry["qa"]['program_re']
            program = program_tokenization(original_program)

    else:
        program = None
    return MathQAExampleSimple(
        source=entry['source'],
        id=this_id,
        question=question,
        question_tokens=tokenize(tokenizer, question),
        facts=facts,
        facts_tokens=facts_tokens,
        options=options,
        answer=answer,
        numbers=numbers,
        number_indices=number_indices,
        original_program=original_program,
        program=program)

