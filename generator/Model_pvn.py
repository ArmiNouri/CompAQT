import wandb
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from random import uniform
from typing import Tuple, List
from config import parameters as conf
import torch.nn.functional as F
from transformers.utils import ModelOutput


# torch.backends.cudnn.enabled=False


if conf.pretrained_model == "bert":
    from transformers import BertModel
elif conf.pretrained_model == "roberta":
    from transformers import RobertaModel
elif conf.pretrained_model == "finbert":
    from transformers import BertModel
elif conf.pretrained_model == "longformer":
    from transformers import LongformerModel


class OperatorOutput(ModelOutput):
    operator_latents = None
    operator_encoded = None
    operator_logits = None
    operator_preds = None
    operator_golds = None
    operator_mask = None
    operator_weights = None
    program_ids = None
    program_mask = None


class OperandOutput(ModelOutput):
    operand1_logits=None
    operand1_preds=None
    operand1_golds=None
    operand2_logits=None
    operand2_preds=None
    operand2_golds=None
    operator_mask=None
    operand_weights=None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        if d_model % 2 == 0: pe[:, 0, 1::2] = torch.cos(position * div_term)
        else: pe[:, 0, 1::2] = torch.cos(position * div_term)[:, :-1]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.shape[1]].transpose(1, 0)
        return self.dropout(x)


class PointerNetwork(nn.Module):
    """
    Adapted from https://hyperscience.com/tech-blog/power-of-pointer-networks/ 
    """
    def __init__(self, embedding_dim: int, num_operands: int, dropout_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_operands = num_operands

        self.lstm_cell = nn.LSTMCell(
            input_size=embedding_dim, hidden_size=embedding_dim
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.reference = nn.Linear(embedding_dim, 1)
        self.decoder_weights = nn.Linear(embedding_dim, embedding_dim)
        self.encoder_weights = nn.Linear(embedding_dim, embedding_dim)

    def attention(
        self, token_embeddings: torch.tensor, hx: torch.tensor
    ) -> torch.tensor:
        batch_size, n_tokens, hidden_dim = token_embeddings.shape

        decoder_query = self.decoder_weights(hx)
        token_embeddings = self.encoder_weights(token_embeddings)

        decoder_query = decoder_query.repeat(
            n_tokens, 1, 1
        )  # n_token x batch_size x embedding_dim
        decoder_query = decoder_query.transpose(
            0, 1
        )  # batch_size x n_token x embedding_dim
        comparison = torch.tanh(decoder_query + token_embeddings)
        probabilities = torch.log_softmax(
            self.reference(comparison).reshape(batch_size, n_tokens), 1
        )
        return probabilities

    def forward(
        self, token_embeddings: torch.tensor, operator_encoded: torch.tensor
    ) -> Tuple[torch.tensor, List[int]]:
        batch_size, _, _ = token_embeddings.shape
        overall_probabilities = []
        batch_identifier = torch.arange(batch_size).type(torch.LongTensor)

        peak_indices = []
        decoder_input = operator_encoded
        for step in range(self.num_operands):
            hx, cx = self.lstm_cell(decoder_input)
            hx = self.dropout(hx)
            probabilities = self.attention(token_embeddings, hx)

            _, peak_idx = probabilities.max(dim=1)
            decoder_input = token_embeddings[batch_identifier, peak_idx, :]

            overall_probabilities.append(probabilities)
            peak_indices.append(peak_idx)

        overall_probabilities = torch.stack(overall_probabilities).transpose(0, 1)
        peak_indices = torch.stack(peak_indices).t()
        return overall_probabilities, peak_indices


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim

    def forward(self, query, key, value):
        x = torch.bmm(query, key.transpose(1, 2))
        weights = torch.softmax(x, dim=self.dim)
        x = torch.bmm(weights, value)
        return x, weights


class OperatorRNN(nn.Module):
    def __init__(self, vocab, start_token, input_len, output_len, hidden_size, output_size, dropout_rate):
        super(OperatorRNN, self).__init__()
        # input: hidden state of verbalized operators generated so far: bs, 768
        # hidden: question-fact attention output: bs, 511, 768
        self.vocab = vocab # 1, 14, 768 
        self.start_token = start_token # 1, 768
        self.output_len = output_len # 21
        self.hidden_size = hidden_size # 768
        self.teacher_forcing_prob = 0.5
        self.in_proj = nn.Linear(input_len, 1)
        # self.pos = PositionalEncoding(hidden_size, 0.0)
        self.attention = Attention(dim=2) # nn.MultiheadAttention(hidden_size, num_heads=1, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        # self.pointer = PointerNetwork(embedding_dim=hidden_size, num_operands=output_len, dropout_rate=dropout_rate)
        self.cell = nn.LSTMCell(hidden_size, hidden_size)
        self.out_prj = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, is_training, qf_encoded, op_encoded):
        # qf_encoded is the encoded question-fact tokens, of shape bs, 511, 768 
        device = qf_encoded.device
        bs, _, _ = list(qf_encoded.size())
        vocab = self.vocab.repeat(bs, 1, 1).to(device) #  bs, 14, 768
        # op_attn, op_weights = self.attention(self.pos(vocab), self.pos(qf_encoded), self.pos(qf_encoded)) # bs, 14, 768
        h = self.in_proj(qf_encoded.transpose(1, 2)).squeeze(2).to(device) # bs, 768
        c = torch.zeros(bs, self.hidden_size).to(device) # bs, 768
        x = self.start_token.repeat(bs, 1).to(device) # bs, 768
        # print(op_attn.shape, self.in_proj.in_features, self.in_proj(op_attn.transpose(1, 2)).transpose(1, 2).shape)
        # probs, indices = self.pointer(vocab, self.in_proj(qf_encoded.transpose(1, 2)).squeeze(2))
        output = []
        for i in range(self.output_len):
            h, c = self.cell(x, (h, c)) # bs, 768
            h = self.dropout(h.unsqueeze(1)) # bs, 768
            output.append(h)
            teach = uniform(0, 1) < self.teacher_forcing_prob
            if is_training and teach: 
                t = op_encoded[:, i, :] # bs, 768
                x, _ = self.attention(x.unsqueeze(1), t.unsqueeze(1), t.unsqueeze(1)) # bs, 1, 768
                x = x.squeeze(1)
            else: 
                x, _ = self.attention(x.unsqueeze(1), h, h) # bs, 1, 768
            x = x.squeeze(1)
            h = h.squeeze(1)
        output = torch.cat(output, dim=1)  # bs, 7, 768
        probs = self.out_prj(output) # bs, 7, 14
        return probs, output # torch.softmax(probs, dim=2) 


class OperatorModel(nn.Module):
    def __init__(self, bert_model, mask_token_id, operator_length, hidden_size, dropout_rate, operator_list, operator_list_ids, device):
        super(OperatorModel, self).__init__()
        self.mask_token_id = mask_token_id
        self.bert = bert_model
        self.operator_length = operator_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.operator_list = operator_list
        self.operator_list_size = len(self.operator_list)
        self.operator_list_ids = operator_list_ids

        all_operator_token_type_ids = torch.zeros((self.operator_list_size, 4), dtype=torch.long).to(device)
        all_operator_position_ids = torch.zeros((self.operator_list_size, 4), dtype=torch.long).to(device)
        operator_vocab = self.bert(input_ids=torch.tensor(operator_list_ids).to(device), token_type_ids=all_operator_token_type_ids, position_ids=all_operator_position_ids) # 13, 4 --> 14, 4, 768 
        # operator_vocab.hidden_states
        # self.operator_vocab = operator_vocab.last_hidden_state[:, 0, :].unsqueeze(1).transpose(0, 1) # pick the CLS token: 14, 1, 768 ==> 1, 14, 768
        self.operator_vocab = torch.sum(operator_vocab.last_hidden_state, dim=1).unsqueeze(1).transpose(0, 1) # pick the CLS token: 14, 1, 768 ==> 1, 14, 768
        self.oprator_vocab = nn.Parameter(self.operator_vocab, requires_grad=False) # 1, 14, 768
        self.operator_start = self.operator_vocab[:, 2, :] # 1, 768

        self.text_pos = PositionalEncoding(hidden_size, 0.0)
        self.operator_pos = PositionalEncoding(self.operator_list_size, 0.0)
        self.question_facts_attn_for_operator = Attention(dim=2) # nn.MultiheadAttention(hidden_size, num_heads=1, dropout=dropout_rate, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        # self.operator_decoder = nn.MultiheadAttention(hidden_size, num_heads=1, dropout=dropout_rate, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        self.operator_dropout = nn.Dropout(dropout_rate)
        self.operator_shrink = nn.Linear(self.hidden_size, self.operator_list_size)
        self.operator_rnn = OperatorRNN(vocab=self.operator_vocab, start_token=self.operator_start, 
                                        input_len=511, output_len=operator_length, 
                                        hidden_size=hidden_size, output_size=self.operator_list_size,
                                        dropout_rate=dropout_rate)

    # masker for decoder
    def generate_square_subsequent_mask(self, input_seq_len: int, output_seq_len: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask = torch.triu(torch.full((input_seq_len, output_seq_len), float('-inf')), diagonal=1)
        return mask

    def forward(self, is_training, question_encoded, question_mask, facts_encoded, facts_mask, distances, program_ids, program_mask, operator_ids, operator_mask, device):        
        batch_size, seq_len, hidden_dim = list(question_encoded.size())
        operator_encoded = [self.operator_vocab.squeeze(0)[operator_ids[i], :].unsqueeze(0) for i in range(batch_size)]
        operator_encoded = torch.cat(operator_encoded, dim=0).to(device) # bs, 7, 768
        
        # question-facts attention for operator prediction
        # bs, 512, 768
        question_facts_latents_for_operator, question_facts_weights_for_operator = self.question_facts_attn_for_operator(self.text_pos(question_encoded), 
                                                                                                                        self.text_pos(facts_encoded), 
                                                                                                                        self.text_pos(facts_encoded)) 
                                                                                                                        # key_padding_mask=facts_mask)
        # augment with distances
        min_distances = torch.min(distances, dim=2).values # bs, 512
        min_distances = min_distances.unsqueeze(2).repeat(1, 1, self.hidden_size) # bs, 512, 768
        x = question_facts_latents_for_operator[:, 1:, :] * min_distances[:, 1:, :] # discard CLS: bs, 511, 768
        x, latents = self.operator_rnn(is_training, x, operator_encoded) # bd, 7, 14 and bs, 7, 768 
        operator_latents = latents # bs, 7, 768
        operator_closest = x  # self.operator_dropout(x) # bs, 7, 14

        # operator-question projection
        operator_indices = torch.arange(self.operator_length)*4
        operator_golds = program_ids[:, operator_indices] # bs, 7
        operator_logits = F.softmax(operator_closest, dim=2) # bs, 7, 14
        operator_preds = torch.argmax(operator_logits, dim=2) # bs, 7

        return OperatorOutput(
                operator_latents=operator_latents, 
                operator_encoded=operator_encoded, 
                operator_logits=operator_logits,
                operator_preds=operator_preds, 
                operator_golds=operator_golds,
                operator_mask=operator_mask.long(),
                program_ids=program_ids, 
                operator_weights=question_facts_weights_for_operator,
                program_mask=program_mask)


class OperandModel(nn.Module):
    def __init__(self, bert_model, mask_token_id, operator_length, hidden_size, dropout_rate, operator_list, operator_list_ids, const_list):
        super(OperandModel, self).__init__()
        self.mask_token_id = mask_token_id
        self.bert = bert_model
        self.operator_length = operator_length
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.operator_list = operator_list
        self.operator_list_size = len(self.operator_list)
        self.operator_list_ids = operator_list_ids
        self.const_list = const_list

        self.pos = PositionalEncoding(hidden_size, 0.0)
        self.const_embed_layer = nn.Embedding(len(self.const_list), hidden_size)
        self.facts_question_attn_for_operand = Attention(dim=2) # nn.MultiheadAttention(hidden_size, num_heads=1, dropout=dropout_rate, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        self.constants_question_attn_for_operand = Attention(dim=2) # nn.MultiheadAttention(hidden_size, num_heads=1, dropout=dropout_rate, kdim=hidden_size, vdim=hidden_size, batch_first=True)
        self.pointer = PointerNetwork(embedding_dim=hidden_size, num_operands=2, dropout_rate=dropout_rate)
        
    def forward(self, is_training, question_encoded, question_mask, facts_encoded, facts_mask, program_ids, program_mask, operator_ids, operator_mask, operator_latents, device):
        # question-facts attention for operand prediction
        # bs, 512, 768
        facts_question_latents_for_operand, facts_question_weights_for_operand = \
                                    self.facts_question_attn_for_operand(self.pos(facts_encoded), 
                                                                        self.pos(question_encoded), 
                                                                        self.pos(question_encoded)) 
                                                                        # key_padding_mask=question_mask)
        # question-constants attention for operand prediction
        batch_size = facts_question_latents_for_operand.shape[0]
        constants = self.const_embed_layer(torch.arange(len(self.const_list)).to(device)) # 31, 768
        constants = constants.unsqueeze(0).repeat(batch_size, 1, 1) # bs, 31, 768
        # bs, 31, 768
        constants_question_latents_for_operand, constants_question_weights_for_operand = \
                                    self.constants_question_attn_for_operand(self.pos(constants), 
                                                                        self.pos(question_encoded), 
                                                                        self.pos(question_encoded)) 
                                                                        # key_padding_mask=question_mask)
        options = torch.cat([constants_question_latents_for_operand, facts_question_latents_for_operand], dim=1) # bs, 31+512, 768
        operand1_logits, operand2_logits, operand1_preds, operand2_preds = [], [], [], []
        for cur_step in range(self.operator_length):
            idx = self.const_list.index('#' + str(cur_step))
            cur_operator = operator_latents[:, cur_step, :] # bs, 768 TODO: add this to the attention
            step_mask = torch.zeros(batch_size, len(self.const_list)-idx, self.hidden_size).to(device) # bs, 31-idx, 768
            cur_options = torch.cat([options[:, :idx, :], step_mask, options[:, len(self.const_list):, :]], dim=1) # bs, 31+512, 768
            probabilities, indices = self.pointer(cur_options, cur_operator)
            operand1_logits.append(probabilities[:, 0, :].unsqueeze(1)) # bs, 1, 31+512
            operand2_logits.append(probabilities[:, 1, :].unsqueeze(1)) # bs, 1, 31+512
            operand1_preds.append(indices[:, 0].unsqueeze(1)) # bs, 1, 1
            operand2_preds.append(indices[:, 1].unsqueeze(1)) # bs, 1, 1
        
        operand1_logits = torch.cat(operand1_logits, dim=1) # bs, 7, 31+512
        operand2_logits = torch.cat(operand2_logits, dim=1) # bs, 7, 31+512
        operand1_preds = torch.cat(operand1_preds, dim=1) # bs, 7, 1
        operand2_preds = torch.cat(operand2_preds, dim=1) # bs, 7, 1
        operator_indices = torch.arange(self.operator_length)*4
        operand1_golds = program_ids[:, operator_indices+1]
        operand2_golds = program_ids[:, operator_indices+2]

        return OperandOutput(
            operand1_logits=operand1_logits,
            operand1_preds=operand1_preds,
            operand1_golds=operand1_golds,
            operand2_logits=operand2_logits,
            operand2_preds=operand2_preds,
            operand2_golds=operand2_golds,
            operator_mask=operator_mask.long(),
            operand_weights=facts_question_weights_for_operand)


class Bert_model(nn.Module):
    def __init__(self, mask_token_id, program_length, sequence_length, hidden_size, dropout_rate, operator_list, operator_list_ids, const_list, device):
        super(Bert_model, self).__init__()

        self.mask_token_id = mask_token_id
        self.program_length = program_length
        self.operator_length = self.program_length//4
        self.operator_list_size = len(operator_list)   # add, subtract, ...
        self.const_list_size = len(const_list)   # const_100, const_10, ...
        # self.reserved_token_size = self.operator_list_size + self.const_list_size
        self.operand_list_size = self.const_list_size + sequence_length
        self.hidden_size = hidden_size
        # self.const_ind = nn.Parameter(torch.arange(0, self.const_list_size), requires_grad=False) ## required grad True or False?

        if conf.pretrained_model == "bert":
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir, output_hidden_states=True)
        elif conf.pretrained_model == "roberta":
            self.bert = RobertaModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir, output_hidden_states=True)
        elif conf.pretrained_model == "finbert":
            self.bert = BertModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir, output_hidden_states=True)
        elif conf.pretrained_model == "longformer":
            self.bert = LongformerModel.from_pretrained(
                conf.model_size, cache_dir=conf.cache_dir, output_hidden_states=True)
        self.bert.to(device)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.operator_model = OperatorModel(self.bert, self.mask_token_id, self.operator_length, hidden_size, dropout_rate, operator_list, operator_list_ids, device=device)
        self.operand_model = OperandModel(self.bert, self.mask_token_id, self.operator_length, hidden_size, dropout_rate, operator_list, operator_list_ids, const_list)

    def dist_matrix(self, a, b, eps=1e-8):
        """# pairwise distance between question tokens and fact tokens
        distances = []
        for i in range(batch_size):
            dist = self.dist_matrix(question_encoded[i], facts_encoded[i]) # 512, 512
            # create a mask where either the question or the fact token is [PAD]
            q_mask = 1. - question_mask[0].long() # 512
            # additionally, mask the [CLS] token in the question
            q_mask[0] = 0.
            f_mask = 1. - facts_mask[0].long() # 512
            mask = torch.outer(q_mask, f_mask) # 512, 512
            dist = dist * mask # 512, 512
            distances.append(dist.unsqueeze(0)) # 1, 512 ,512
        distances = torch.cat(distances, dim=0) # bs, 512, 512
        Adapted from https://stackoverflow.com/a/67588366
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        dist_mt = 1.0 - sim_mt
        return (dist_mt+1)/2 # 512, 512

    def forward(self, is_training, question_ids, question_mask, facts_ids, facts_mask, numbers_mask, program_ids, program_mask, operator_ids, operator_mask, device):        
        # question encoder
        question_token_type_ids = torch.zeros_like(question_ids, dtype=torch.long, device=device)
        question_bert_outputs = self.bert(input_ids=question_ids.long(), attention_mask=question_mask, token_type_ids=question_token_type_ids)
        question_bert_sequence_output = question_bert_outputs.last_hidden_state
        batch_size, seq_length, bert_dim = list(question_bert_sequence_output.size())
        device = question_bert_sequence_output.device
        question_encoded = question_bert_sequence_output

        # facts encoder
        facts_token_type_ids = torch.zeros_like(facts_ids, dtype=torch.long, device=device)
        facts_bert_outputs = self.bert(input_ids=facts_ids.long(), attention_mask=facts_mask, token_type_ids=facts_token_type_ids)
        facts_bert_sequence_output = facts_bert_outputs.last_hidden_state
        facts_encoded = facts_bert_sequence_output

        # pairwise distance between question tokens and fact tokens
        distances = []
        for i in range(batch_size):
            dist = self.dist_matrix(question_encoded[i], facts_encoded[i]) # 512, 512
            distances.append(dist.unsqueeze(0)) # 1, 512, 412
        distances = torch.cat(distances, dim=0) # bs, 512, 512 

        distances_masked = []
        for i in range(batch_size):
            # dist = self.dist_matrix(question_encoded[i], facts_encoded[i]) # 512, 512
            dist = distances[i] # 512, 512
            # create a mask where either the question or the fact token is [PAD]
            q_mask = 1. - question_mask[0].long() # 512
            # additionally, mask the [CLS] token in the question
            q_mask[0] = 0.
            f_mask = 1. - facts_mask[0].long() # 512
            mask = torch.outer(q_mask, f_mask) # 512, 512
            dist = dist * mask # 512, 512
            distances_masked.append(dist.unsqueeze(0)) # 1, 512 ,512
        distances_masked = torch.cat(distances_masked, dim=0) # bs, 512, 512

        # operator and operand predictions
        operator_output = self.operator_model(is_training, question_encoded, question_mask, facts_encoded, facts_mask, distances, program_ids, program_mask, operator_ids, operator_mask, device)
        operands_output = self.operand_model(is_training, question_encoded, question_mask, facts_encoded, facts_mask, program_ids, program_mask, operator_ids, operator_mask, operator_output.operator_latents, device)
        
        return distances_masked, operator_output, operands_output
