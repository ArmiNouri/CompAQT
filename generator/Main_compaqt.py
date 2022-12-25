#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script
"""
from tqdm import tqdm
import json
import os
from datetime import datetime
import time
import logging
from utils import *
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim
import numpy as np


from Model_compaqt import Bert_model
from CounterComp import Sampler, triplet_margin_with_distance_loss

if conf.pretrained_model == "bert":
    print("Using bert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "roberta":
    print("Using roberta")
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "finbert":
    print("Using finbert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_config = BertConfig.from_pretrained(conf.model_size)

elif conf.pretrained_model == "longformer":
    print("Using longformer")
    from transformers import LongformerTokenizer, LongformerConfig
    tokenizer = LongformerTokenizer.from_pretrained(conf.model_size)
    model_config = LongformerConfig.from_pretrained(conf.model_size)


# create output paths
if conf.mode == "train":
    model_dir_name = conf.model_save_name + "_" + \
        datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(conf.output_path, model_dir_name)
    results_path = os.path.join(model_dir, "results")
    saved_model_path = os.path.join(model_dir, "saved_model")
    os.makedirs(saved_model_path, exist_ok=False)
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

else:
    saved_model_path = os.path.join(conf.output_path, conf.saved_model_path)
    model_dir_name = datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = os.path.join(
        conf.output_path, 'inference_only_' + model_dir_name)
    results_path = os.path.join(model_dir, "results")
    os.makedirs(results_path, exist_ok=False)
    log_file = os.path.join(results_path, 'log.txt')

op_list = read_txt(conf.op_list_file, log_file)
operation_list = op_list
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

print(op_list)
print(const_list)

train_data, train_examples, op_list, const_list = \
    read_examples(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file, source=conf.source)

sampler = Sampler(operation_list, train_examples)

valid_data, valid_examples, op_list, const_list = \
    read_examples(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file, source=conf.source)

test_data, test_examples, op_list, const_list = \
    read_examples(input_path=conf.test_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file, source=conf.source)

kwargs = {"examples": train_examples,
          "tokenizer": tokenizer,
          "max_seq_length": conf.max_seq_length,
          "max_program_length": conf.max_program_length,
          "is_training": True,
          "op_list": op_list,
          "op_list_size": len(op_list),
          "const_list": const_list,
          "const_list_size": len(const_list),
          "verbose": True}

train_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = valid_examples
kwargs["is_training"] = False
valid_features = convert_examples_to_features(**kwargs)
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tensorize_sample(x):
    input_ids = torch.tensor(x['input_ids']).to(conf.device)
    input_mask = torch.tensor(x['input_mask']).to(conf.device)
    segment_ids = torch.tensor(x['segment_ids']).to(conf.device)
    program_ids = torch.tensor(x['program_ids']).to(conf.device)
    program_mask = torch.tensor(x['program_mask']).to(conf.device)
    option_mask = torch.tensor(x['option_mask']).to(conf.device)
    return input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask


def get_alignment_loss(this_loss, all_weights, distances, input_mask):
    #############compositional loss################
    this_loss = (1.0-conf.lmbda) * this_loss
    start_weights = torch.zeros_like(all_weights[0])
    all_weights = [start_weights] + all_weights
    for i in range(distances.size(0)): distances[i].fill_diagonal_(1.0)
    for i in range(1, len(all_weights)):
        weights_i = all_weights[i]
        weights_i_min_1 = all_weights[i-1]
        bs, l, _ = list(weights_i.size())
        distances = distances + (1.0-input_mask.unsqueeze(2).repeat(1, 1, l))
        distances = torch.min(distances, dim=1).values
        distances = distances * input_mask
        distances = distances.unsqueeze(2).repeat(1, 1, l)
        p_prior = torch.nn.functional.relu(distances - weights_i_min_1)
        comp_loss = weights_i - p_prior
        comp_loss = comp_loss ** 2
        comp_loss = comp_loss.sum(dim=2)
        comp_loss = comp_loss.sum(dim=1) / l
        comp_loss = comp_loss.sum() / bs
        comp_loss = comp_loss - conf.alpha * i
        this_loss += (conf.lmbda/len(all_weights)) * comp_loss
    ###############################################
    return this_loss


def get_countercomp_triplet_loss(this_loss, train_iterator, x, a_decoder_hidden, model):
    ### Begin CounterComp sampling ###
            example_idxs = x['example_index']
            sample_indices = [sampler.sample_pos_neg(train_examples[idx]) for idx in example_idxs]
            sample_indices = [[x[0], x[1], x[2], x[3], x[4]] for x in sample_indices]
            anchors = train_iterator.get_items_at_indices([x[0] for x in sample_indices])
            poss = train_iterator.get_items_at_indices([x[1] for x in sample_indices])
            negs = train_iterator.get_items_at_indices([x[2] for x in sample_indices])
            pos_dists = torch.tensor([x[3] for x in sample_indices], device=conf.device)
            neg_dists = torch.tensor([x[4] for x in sample_indices], device=conf.device)
            margin = (neg_dists-pos_dists+1.0)/2.0   # Token level distances, normalized between 0 and 1.
            margin = margin.unsqueeze(1)
            a_input_ids, a_input_mask, a_segment_ids, a_program_ids, a_program_mask, a_option_mask = tensorize_sample(anchors)
            p_input_ids, p_input_mask, p_segment_ids, p_program_ids, p_program_mask, p_option_mask = tensorize_sample(poss)
            n_input_ids, n_input_mask, n_segment_ids, n_program_ids, n_program_mask, n_option_mask = tensorize_sample(negs)
            # a_weights, a_distances, a_decoder_hidden, a_logits = model(True, a_input_ids, a_input_mask, a_segment_ids,
            #                     a_option_mask, a_program_ids, a_program_mask, device=conf.device)
            p_weights, p_distances, p_decoder_hidden, p_logits = model(True, p_input_ids, p_input_mask, p_segment_ids,
                                p_option_mask, p_program_ids, p_program_mask, device=conf.device)
            n_weights, n_distances, n_decoder_hidden, n_logits = model(True, n_input_ids, n_input_mask, n_segment_ids,
                                n_option_mask, n_program_ids, n_program_mask, device=conf.device)
            triplet_loss = triplet_margin_with_distance_loss(a_decoder_hidden, p_decoder_hidden, n_decoder_hidden, margin)
            ### End CounterComp sampling ###
            return (1.-conf.beta) * this_loss + conf.beta * triplet_loss


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    model = Bert_model(num_decoder_layers=conf.num_decoder_layers,
                       hidden_size=model_config.hidden_size,
                       dropout_rate=conf.dropout_rate,
                       program_length=conf.max_program_length,
                       input_length=conf.max_seq_length,
                       op_list=op_list,
                       const_list=const_list)

    print('Number of trainable parameters in model:', count_parameters(model))
    # model = nn.DataParallel(model)
    model.to(conf.device)

    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    # torch.autograd.set_detect_anomaly(True)

    train_iterator = DataLoader(
        is_training=True, data=train_features, batch_size=conf.batch_size, reserved_token_size=reserved_token_size, shuffle=True)

    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0

    for _ in range(conf.epoch):
        train_iterator.reset()
        for x in train_iterator:

            input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask = tensorize_sample(x)

            model.zero_grad()
            optimizer.zero_grad()

            all_weights, distances, decoder_hidden, this_logits = model(True, input_ids, input_mask, segment_ids,
                                option_mask, program_ids, program_mask, device=conf.device)

            this_loss = criterion(
                this_logits.view(-1, this_logits.shape[-1]), program_ids.view(-1))

            this_loss = this_loss * program_mask.view(-1)
            # per token loss
            this_loss = this_loss.sum() / program_mask.sum()

            if conf.compaqt: this_loss = get_alignment_loss(this_loss, all_weights, distances, input_mask)
            if conf.countercomp: this_loss = get_countercomp_triplet_loss(this_loss, train_iterator, x, decoder_hidden, model)

            print(this_loss)
            record_loss += this_loss.item()
            record_k += 1
            k += 1
            
            this_loss.backward()
            optimizer.step()

            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f" %
                          (k, record_loss / record_k))
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % conf.report == 0:
                print("Round: ", k / conf.report)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // conf.report, cost_time))
                start_time = time.time()
                if k // conf.report >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(
                        saved_model_path, 'loads', str(k // conf.report))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    torch.save(model.state_dict(),
                               saved_model_path_cnt + "/model.pt")

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // conf.report))
                    os.makedirs(results_path_cnt, exist_ok=True)
                    validation_result = evaluate(
                        valid_examples, valid_features, model, results_path_cnt, 'valid')
                    # write_log(log_file, validation_result)

                model.train()


def evaluate(data_ori, data, model, ksave_dir, mode='valid'):

    pred_list = []
    pred_unk = []

    ksave_dir_mode = os.path.join(ksave_dir, mode)
    os.makedirs(ksave_dir_mode, exist_ok=True)

    data_iterator = DataLoader(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)

    k = 0
    all_results = []
    with torch.no_grad():
        for x in tqdm(data_iterator):

            input_ids = x['input_ids']
            input_mask = x['input_mask']
            segment_ids = x['segment_ids']
            program_ids = x['program_ids']
            program_mask = x['program_mask']
            option_mask = x['option_mask']

            ori_len = len(input_ids)
            for each_item in [input_ids, input_mask, segment_ids, program_ids, program_mask, option_mask]:
                if ori_len < conf.batch_size_test:
                    each_len = len(each_item[0])
                    pad_x = [0] * each_len
                    each_item += [pad_x] * (conf.batch_size_test - ori_len)

            input_ids = torch.tensor(input_ids).to(conf.device)
            input_mask = torch.tensor(input_mask).to(conf.device)
            segment_ids = torch.tensor(segment_ids).to(conf.device)
            program_ids = torch.tensor(program_ids).to(conf.device)
            program_mask = torch.tensor(program_mask).to(conf.device)
            option_mask = torch.tensor(option_mask).to(conf.device)

            _, _, _, logits = model(False, input_ids, input_mask,
                           segment_ids, option_mask, program_ids, program_mask, device=conf.device)

            for this_logit, this_id in zip(logits.tolist(), x["unique_id"]):
                all_results.append(
                    RawResult(
                        unique_id=int(this_id),
                        logits=this_logit,
                        loss=None
                    ))

    output_prediction_file = os.path.join(ksave_dir_mode,
                                          "predictions.json")
    output_nbest_file = os.path.join(ksave_dir_mode,
                                     "nbest_predictions.json")
    output_eval_file = os.path.join(ksave_dir_mode, "full_results.json")
    output_error_file = os.path.join(ksave_dir_mode, "full_results_error.json")

    all_predictions, all_nbest = compute_predictions(
        data_ori,
        data,
        all_results,
        n_best_size=conf.n_best_size,
        max_program_length=conf.max_program_length,
        tokenizer=tokenizer,
        op_list=op_list,
        op_list_size=len(op_list),
        const_list=const_list,
        const_list_size=len(const_list))
    write_predictions(all_predictions, output_prediction_file)
    write_predictions(all_nbest, output_nbest_file)

    if mode == "valid":
        original_file = conf.valid_file
    else:
        original_file = conf.test_file

    exe_acc, prog_acc = evaluate_result(
        output_nbest_file, original_file, output_eval_file, output_error_file, program_mode=conf.program_mode)

    prog_res = "exe acc: " + str(exe_acc) + " prog acc: " + str(prog_acc)
    write_log(log_file, prog_res)


    return


if __name__ == '__main__':
    if conf.mode == "train":
        train()
