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
from finqa_utils import verbalization_map
from config import parameters as conf
from torch import nn
import torch
import torch.optim as optim


from generator.Model_pvn import Bert_model

if conf.pretrained_model == "bert":
    print("Using bert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained(conf.model_size)
    model_config = BertConfig.from_pretrained(conf.model_size, output_hidden_state=True)

elif conf.pretrained_model == "roberta":
    print("Using roberta")
    from transformers import RobertaTokenizer
    from transformers import RobertaConfig
    tokenizer = RobertaTokenizer.from_pretrained(conf.model_size)
    model_config = RobertaConfig.from_pretrained(conf.model_size, output_hidden_state=True)

elif conf.pretrained_model == "finbert":
    print("Using finbert")
    from transformers import BertTokenizer
    from transformers import BertConfig
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model_config = BertConfig.from_pretrained(conf.model_size, output_hidden_state=True)

elif conf.pretrained_model == "longformer":
    print("Using longformer")
    from transformers import LongformerTokenizer, LongformerConfig
    tokenizer = LongformerTokenizer.from_pretrained(conf.model_size)
    model_config = LongformerConfig.from_pretrained(conf.model_size, output_hidden_state=True)


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
op_list = [op + '(' for op in op_list]
op_list = ['EOF', 'UNK', 'GO', ')'] + op_list
op_list_ids = [tokenizer.tokenize(verbalization_map[x.replace('(', '')]) for x in op_list]
op_list_ids = [[tokenizer.cls_token] + x + ['[PAD]']*(3-len(x)) for x in op_list_ids]
op_list_ids = [tokenizer.convert_tokens_to_ids(x) for x in op_list_ids]

const_list = read_txt(conf.const_list_file, log_file)
const_list = [const.lower().replace('.', '_') for const in const_list]
reserved_token_size = len(op_list) + len(const_list)

print(op_list)
print(const_list)

train_data, train_examples, op_list, const_list = \
    read_examples_simple(input_path=conf.train_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file, source=conf.source)

valid_data, valid_examples, op_list, const_list = \
    read_examples_simple(input_path=conf.valid_file, tokenizer=tokenizer,
                  op_list=op_list, const_list=const_list, log_file=log_file, source=conf.source)

test_data, test_examples, op_list, const_list = \
    read_examples_simple(input_path=conf.test_file, tokenizer=tokenizer,
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
valid_features = convert_examples_to_features(**kwargs)
kwargs["is_training"] = False
kwargs["examples"] = test_examples
test_features = convert_examples_to_features(**kwargs)


def get_operator_loss(operator_output, q_f_dists):
    def loss_per_step(idx, alpha=0.01):
        mask = fqo.program_mask[:, idx*4]  # bs
        if mask.sum().item() == 0.: return torch.tensor(0., requires_grad=True).to(mask.device) 

        loss3 = F.cross_entropy(fqo.operator_logits[:, idx, :], 
                                fqo.operator_golds[:, idx],
                                reduction='none') # bs
        loss3 = loss3 * mask
        loss3 = loss3.sum() / mask.sum()

        return loss3 # + alpha * idx

    def guided_attn_reg():
        # q_f_dists: bs, 512, 512
        bs, z_size, u_size = q_f_dists.shape
        # p_prior = q_f_dists / q_f_dists.sum(dim=2).unsqueeze(2).repeat(1, 1, u_size) # bs, 512, 512
        # p_prior = torch.nan_to_num(p_prior, nan=0.0) # for when the denominator is 0.0 due to masking
        p_att = fqo.operator_weights # bs, 512, 512
        err = (p_att-q_f_dists)**2 # bs, 512, 512
        output = 1/z_size * err.sum(dim=2).sum(dim=1) # first sum along u (fact tokens) then along z (question tokens)
        return output.sum()

    fqo = operator_output
    operator_length = conf.max_program_length//4    
    bs, sl, hd = list(fqo.operator_latents.size())
    mask = fqo.operator_mask
    mask = mask.reshape(bs*sl)

    loss1 = F.mse_loss(fqo.operator_latents.reshape(bs*sl, hd), 
                       fqo.operator_encoded.reshape(bs*sl, hd),
                       reduction='none') # bs, 768
    loss1 = torch.sum(loss1, dim=1)/hd # bs
    loss1 = loss1 * mask
    loss1 = loss1.sum() / bs

    
    inp1 = fqo.operator_latents.reshape(bs*sl, hd)
    inp2 = fqo.operator_encoded.reshape(bs*sl, hd)
    loss2 = F.cosine_embedding_loss(input1=inp1, 
                                    input2=inp2, 
                                    target=torch.ones(bs*sl).to(mask.device),
                                    reduction='none')
    loss2 = loss2 * mask
    loss2 = loss2.sum() / bs

    loss3 = F.cross_entropy(fqo.operator_logits.view(bs*sl, -1), 
                            fqo.operator_golds.view(bs*sl),
                            reduction='none') # bs
    loss3 = loss3 * mask
    loss3 = loss3.sum() / bs

    return loss1 + loss2 + loss3 # / fqo.operator_mask.sum()


def get_operand_loss(operands_output, q_f_dists):
    def loss_per_step(idx, alpha=0.01):
        mask = fqo.operator_mask[:, idx]  # bs
        if mask.sum().item() == 0.: return torch.tensor(0., requires_grad=True).to(mask.device) 
        loss1 = F.cross_entropy(fqo.operand1_logits[:, idx, :].squeeze(1), fqo.operand1_golds[:, idx], reduction='none') # bs
        loss1 = loss1 * mask
        loss1 = loss1.sum()
        loss2 = F.cross_entropy(fqo.operand2_logits[:, idx, :].squeeze(1), fqo.operand2_golds[:, idx], reduction='none') # bs
        loss2 = loss2 * mask
        loss2 = loss2.sum()
        return loss1 + loss2 # + alpha * idx        # loss1 = loss1 - alpha*idx

    def guided_attn_reg():
        # q_f_dists: bs, 512, 512
        bs, z_size, u_size = q_f_dists.shape
        p_prior = q_f_dists / q_f_dists.sum(dim=1).unsqueeze(1).repeat(1, z_size, 1) # bs, 512, 512
        p_prior = torch.nan_to_num(p_prior, nan=0.0) # for when the denominator is 0.0 due to masking
        p_prior = p_prior.transpose(1, 2) # swapping the z (question) and u (fact) dimensions, so they match the order of dims in weights
        p_att = fqo.operand_weights # bs, 512, 512
        err = (p_att-p_prior)**2 # bs, 512, 512
        output = 1/u_size * err.sum(dim=2).sum(dim=1) # first sum along z (question tokens) then along u (fact tokens)
        return output.mean() 

    fqo = operands_output
    operator_length = conf.max_program_length//4
    loss = torch.stack([loss_per_step(idx) for idx in range(operator_length)], dim=0)
    return loss.sum() / fqo.operator_mask.sum() + guided_attn_reg()


def get_length_loss(operator_output):
    preds = torch.count_nonzero(operator_output.operator_preds, dim=1).float() # bs
    golds = torch.count_nonzero(operator_output.operator_golds, dim=1).float() # bs
    return ((golds - preds)**2).mean()


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


def train():
    # keep track of all input parameters
    write_log(log_file, "####################INPUT PARAMETERS###################")
    for attr in conf.__dict__:
        value = conf.__dict__[attr]
        write_log(log_file, attr + " = " + str(value))
    write_log(log_file, "#######################################################")

    model = Bert_model(mask_token_id=tokenizer.mask_token_id, program_length=conf.max_program_length, sequence_length=conf.max_seq_length,
                        hidden_size=model_config.hidden_size, dropout_rate=conf.dropout_rate,
                        operator_list=op_list, operator_list_ids=op_list_ids, const_list=const_list, device=conf.device)

    model = nn.DataParallel(model)
    device = conf.device
    # device = "cpu"
    model.to(device)
    optimizer = optim.Adam(model.parameters(), conf.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    # torch.autograd.set_detect_anomaly(True)

    train_iterator = DataLoaderSimple(
        is_training=True, data=train_features, batch_size=conf.batch_size, reserved_token_size=reserved_token_size, shuffle=True)

    k = 0
    record_k = 0
    record_loss_k = 0
    loss, start_time = 0.0, time.time()
    record_loss = 0.0
    size = len(train_iterator)

    print("There are ", size, "iterations in the training set.")

    for epoch in range(conf.epoch):
        train_iterator.reset()
        for x in train_iterator:
            model.zero_grad()
            optimizer.zero_grad()
            
            question_ids = torch.tensor(x['question_ids']).to(device)
            question_mask = torch.tensor(x['question_mask']).to(device)
            facts_ids = torch.tensor(x['facts_ids']).to(device)
            facts_mask = torch.tensor(x['facts_mask']).to(device)
            numbers_mask = torch.tensor(x['numbers_mask']).to(device)
            program_ids = torch.tensor(x['program_ids']).to(device)
            program_mask = torch.tensor(x['program_mask']).to(device)
            operator_ids = torch.tensor(x['operator_ids']).to(device)
            operator_mask = torch.tensor(x['operator_mask']).to(device)

            distances, operator_output, operands_output = \
                            model(is_training=True, question_ids=question_ids, question_mask=question_mask, 
                            facts_ids=facts_ids, facts_mask=facts_mask, numbers_mask=numbers_mask, 
                            program_ids=program_ids, program_mask=program_mask, operator_ids=operator_ids, 
                            operator_mask=operator_mask, device=device)
            
            operator_loss = get_operator_loss(operator_output, distances)
            operands_loss = get_operand_loss(operands_output, distances)
            loss = operator_loss + operands_loss 

            if conf.compaqt: loss = get_alignment_loss(loss, operator_output.operator_weights, distances, question_mask)   

            loss.backward()
            optimizer.step()

            record_loss += loss.item()
            record_k += 1
            k += 1

            if k > 1 and k % conf.report_loss == 0:
                write_log(log_file, "%d : loss = %.3f" %
                          (k, record_loss / record_k))
                record_loss = 0.0
                record_k = 0

            if k > 1 and k % size == 0:
                print("Round: ", k / size)
                model.eval()
                cost_time = time.time() - start_time
                write_log(log_file, "%d : time = %.3f " %
                          (k // size, cost_time))
                start_time = time.time()
                if k // size >= 1:
                    print("Val test")
                    # save model
                    saved_model_path_cnt = os.path.join(
                        saved_model_path, 'loads', str(k // size))
                    os.makedirs(saved_model_path_cnt, exist_ok=True)
                    torch.save(model.state_dict(),
                               saved_model_path_cnt + "/model.pt")

                    results_path_cnt = os.path.join(
                        results_path, 'loads', str(k // size))
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
    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    data_iterator = DataLoaderSimple(
        is_training=False, data=data, batch_size=conf.batch_size_test, reserved_token_size=reserved_token_size, shuffle=False)

    k = 0
    total_size = len(data_iterator)
    accurate_count = 0
    device = conf.device
    with torch.no_grad():
        for x in tqdm(data_iterator):

            question_ids = torch.tensor(x['question_ids']).to(device)
            question_mask = torch.tensor(x['question_mask']).to(device)
            facts_ids = torch.tensor(x['facts_ids']).to(device)
            facts_mask = torch.tensor(x['facts_mask']).to(device)
            numbers_mask = torch.tensor(x['numbers_mask']).to(device)
            program_ids = torch.tensor(x['program_ids']).to(device)
            program_mask = torch.tensor(x['program_mask']).to(device)
            operator_ids = torch.tensor(x['operator_ids']).to(device)
            operator_mask = torch.tensor(x['operator_mask']).to(device)

            distances, operator_output, operands_output = \
                            model(is_training=False, question_ids=question_ids, question_mask=question_mask, 
                            facts_ids=facts_ids, facts_mask=facts_mask, numbers_mask=numbers_mask, 
                            program_ids=program_ids, program_mask=program_mask, operator_ids=operator_ids, 
                            operator_mask=operator_mask, device=device)

            operator_preds = operator_output.operator_preds
            operator_preds = [','.join([str(y) for y in z]) for z in operator_preds.tolist()]
            operator_golds = [','.join([str(y) for y in z]) for z in operator_output.operator_golds.tolist()]

            options = [const_list + l for l in x['facts_tokens']]
            operand1_preds = [','.join([options[i][y] for y in z]) for i, z in enumerate(operands_output.operand1_preds.tolist())]
            operand1_golds = [','.join([options[i][y] for y in z]) for i, z in enumerate(operands_output.operand1_golds.tolist())]
            operand2_preds = [','.join([options[i][y] for y in z]) for i, z in enumerate(operands_output.operand2_preds.tolist())]
            operand2_golds = [','.join([options[i][y] for y in z]) for i, z in enumerate(operands_output.operand2_golds.tolist())]

            i = 0
            for z, y in zip(operator_preds, operator_golds):
                print('operator pred vs gold', z, '|', y)  # , '|', operand1_preds[i], '|', operand1_golds[i], '|', operand2_preds[i], '|', operand2_golds[i])
                i += 1
            accurate_count += len([z for z,y in zip(operator_preds, operator_golds) if ',0' in z and z[:z.index(',0')] == y[:y.index(',0')]])
            # accurate_count += len([z for z,y in zip(operator_preds, operator_golds) if z == y])

            # weights = torch.argmax(operator_output.operator_weights, dim=1)
            # for i in range(len(x['question_tokens'])):
            #     print(x['question_tokens'][i][weights[i][0]], x['question_tokens'][i][weights[i][1]])

    print('accuracy:', accurate_count, len(data), accurate_count / len(data))

    return True


if __name__ == '__main__':
    if conf.mode == "train":
        train()