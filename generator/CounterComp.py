from typing import Callable, Optional
import torch
from collections import defaultdict
from random import choice, shuffle


class Sampler:
    def __init__(self, op_list, dataset):
        self.op_list = op_list
        self.id_to_idx = {data.id: idx for idx, data in enumerate(dataset)}
        self.operator_map = self.build_operator_index(dataset)
        self.keys = [k for k in self.operator_map.keys()]

    def extract_operators(self, program):
        steps = program.split('), ')
        ops = [step.split('(')[0].strip() for step in steps]
        return ','.join(ops)

    def build_operator_index(self, dataset):
        operator_map = defaultdict(list)
        for data in dataset:
            ops = self.extract_operators(data.original_program)
            operator_map[ops].append(data)
        return operator_map

    def one_step_perturbation(self, ops):
        output = []
        locations = []
        for idx, op in enumerate(ops):
            # deletion
            if idx > 0 and idx < len(ops)-1: 
                output.append(ops[:idx] + ops[idx+1:])
                locations.append(idx)
            # insertion
            for operator in self.op_list:
                output.append(ops[:idx] + [operator] + ops[idx:])
                locations.append(idx)
                # edit
                if operator != op: 
                    output.append(ops[:idx] + [operator] + ops[idx+1:])
                    locations.append(idx)
        return output, locations

    def get_edit_dist(self, tks1, tks2):
        m, n = len(tks1), len(tks2)
        # Create a table to store results of subproblems
        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
    
        # Fill d[][] in bottom up manner
        for i in range(m + 1):
            for j in range(n + 1):
                # If first string is empty, only option is to
                # insert all characters of second string
                if i == 0:
                    dp[i][j] = j    # Min. operations = j
                # If second string is empty, only option is to
                # remove all characters of second string
                elif j == 0:
                    dp[i][j] = i    # Min. operations = i
                # If last characters are same, ignore last char
                # and recur for remaining string
                elif tks1[i-1] == tks2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                # If last character are different, consider all
                # possibilities and find minimum
                else:
                    dp[i][j] = 1 + min(dp[i][j-1],  # Insert
                                    dp[i-1][j],     # Remove
                                    dp[i-1][j-1])   # Replace
        return dp[m][n]/max([m, n])

    def get_spans(self, sample):
        if sample is None: return []
        tokens = sample.question_tokens
        idx = tokens.index('[SEP]')
        question_tokens = tokens[:idx]
        facts_tokens = tokens[idx+1:]
        ft = set(facts_tokens)
        spans = []
        cur_span = []
        for idx, tk in enumerate(question_tokens):
            if tk not in ft:
                cur_span.append(idx)
            if tk in ft and len(cur_span) > 0:
                spans.append([cur_span[0], idx-1])
                cur_span = []
        if len(cur_span) > 0:
            spans.append([cur_span[0], cur_span[-1]])
        return spans

    def get_tks_by_idx(self, spans, tokens):
        output = []
        for span in spans:
            output = output + tokens[span[0]:span[1]+1]
        return output

    def sample_pos_neg(self, anchor):
        pos = self.sample_pos(anchor)
        neg, loc = self.sample_neg(anchor)
        anchor_spans = self.get_spans(anchor)
        pos_spans = self.get_spans(pos)
        neg_spans = self.get_spans(neg)
        anchor_tks = self.get_tks_by_idx(anchor_spans, anchor.question_tokens)
        pos_tks = self.get_tks_by_idx(pos_spans, pos.question_tokens) if pos is not None else []
        neg_tks = self.get_tks_by_idx(neg_spans, neg.question_tokens) if neg is not None else []
        anchor_idx = self.id_to_idx[anchor.id]
        pos_dist = self.get_edit_dist(anchor_tks, pos_tks) if pos is not None else 0
        neg_dist = self.get_edit_dist(anchor_tks, neg_tks) if neg is not None else 0
        pos_idx = self.id_to_idx[pos.id] if pos is not None else anchor_idx
        neg_idx = self.id_to_idx[neg.id] if neg is not None else anchor_idx
        return anchor_idx, pos_idx, neg_idx, loc, pos_dist, neg_dist

    def sample_pos(self, anchor):
        op = self.extract_operators(anchor.original_program)
        samples = self.operator_map.get(op, [])
        samples = [sample for sample in samples if sample.id != anchor.id]
        if len(samples) == 0:
            return None
        return choice(samples)

    def sample_neg(self, anchor):
        op = self.extract_operators(anchor.original_program)
        perturbeds, locations = self.one_step_perturbation(op.split(','))
        zipped = list(zip(perturbeds, locations))
        shuffle(zipped)
        for perturbed, location in zipped:
            o = ','.join(perturbed)
            if o not in self.operator_map: continue
            return choice(self.operator_map[o]), location
        return None, 0


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")


def _apply_loss_reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss

# Pure Python impl - don't register decomp and don't add a ref.  Defined as a
# helper here since triplet_margin_loss can be nicely implemented with it.
def triplet_margin_with_distance_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: torch.Tensor,
    distance_function: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    # margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    _check_reduction_value(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    assert a_dim == p_dim
    assert p_dim == n_dim

    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    loss = torch.clamp_min(margin + dist_pos - dist_neg, 0)
    return _apply_loss_reduction(loss, reduction)
        