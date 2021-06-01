# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import sys
from .initialize import get_data_parallel_world_size, get_data_parallel_rank, get_data_parallel_group, get_model_parallel_rank, get_model_parallel_world_size

class BaseLayer(nn.Module):

    def __init__(self, config, BaseSublayer, init_method, output_layer_init_method):
        super().__init__()
        self.num_workers = get_data_parallel_world_size()
        self.num_mp = get_model_parallel_world_size()
        self.world_size = torch.distributed.get_world_size()
        self.model_id = get_model_parallel_rank()

        expert_centroids = torch.empty(self.num_workers, config.d_model)
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.expert_network = BaseSublayer(config, init_method, output_layer_init_method=output_layer_init_method)
        self.expert_id = get_data_parallel_rank()
        # self.shuffle = config.base_shuffle
        self.shuffle = True
        self.cpp = self.load_assignment()

        # Add a special attribute to the expert parameters, so we know not to sync their gradients
        for param in self.expert_network.parameters():
            param.expert = True

    def forward(self, input_features, *args, **kwargs):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = True

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            chunk = features.size(0) // self.num_workers
            idxs = list(range(features.size(0)))
            idxs = idxs[get_data_parallel_rank()*chunk:]+idxs[:get_data_parallel_rank()*chunk]

            shuffle_sort = torch.LongTensor(idxs).cuda()
            features = All2All.apply(features[shuffle_sort], None, None, get_data_parallel_group())

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            norm_features = self.expert_network.layer_norm(features)
            token_expert_affinities = norm_features.matmul(self.expert_centroids.transpose(0, 1))

        # Compute which token goes to which expert
        sort_by_expert, input_splits, output_splits = self.balanced_assignment(token_expert_affinities)
       
        # Swap these tokens for the right ones for our expert
        routed_features = All2All.apply(features[sort_by_expert], output_splits, input_splits, get_data_parallel_group())

        if routed_features.size(0) > 0:
            # Mix in the expert network based on how appropriate it is for these tokens
            # norm_features = self.expert_network.layer_norm(routed_features)
            # alpha = torch.sigmoid(norm_features.mv(self.expert_centroids[self.expert_id]) ).unsqueeze(1)
			# xs + self.ff2(self.activation_fn(self.ff1(self.norm(xs))))
            routed_features = self.expert_network(routed_features)
            # routed_features = self.expert_network(routed_features)
        # Return to original worker and ordering
        result = All2All.apply(routed_features, input_splits, output_splits, get_data_parallel_group())[self.inverse_sort(sort_by_expert)]

        if self.shuffle and is_training:
            # Undo shuffling
            result = All2All.apply(result, None, None, get_data_parallel_group())[self.inverse_sort(shuffle_sort)]

        # Return additional Nones for compatibility with TransformerDecoderLayer
        return result.view(input_features.size())

    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return self.cpp.balanced_assignment(scores), None, None

    # Assigns each token to the top k experts
    def greedy_assignment(self, scores, k=1):
        assert False # Not adapted to model parallel

        token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
        token_to_workers, sort_ordering = torch.sort(token_to_workers)
        worker2token = sort_ordering // k

        # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
        output_splits = torch.zeros((self.num_workers,), dtype=torch.long, device=scores.device)
        workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
        output_splits[workers] = counts
        # Tell other workers how many tokens to expect from us
        input_splits = All2All.apply(output_splits)
        return worker2token, input_splits.tolist(), output_splits.tolist()

    def load_assignment(self):
        try:
            import libbase

            return libbase

        except ImportError as e:
            sys.stderr.write(
                "ERROR: missing libbase. run `python setup.py build_ext --inplace`\n"
            )
            raise e


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    '''
    >>> input
    tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
    tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
    tensor([20, 21, 22, 23, 24])                                     # Rank 2
    tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
    >>> input_splits
    [2, 2, 1, 1]                                                     # Rank 0
    [3, 2, 2, 2]                                                     # Rank 1
    [2, 1, 1, 1]                                                     # Rank 2
    [2, 2, 2, 1]                                                     # Rank 3
    >>> output_splits
    [2, 3, 2, 2]                                                     # Rank 0
    [2, 2, 1, 2]                                                     # Rank 1
    [1, 2, 1, 2]                                                     # Rank 2
    [1, 2, 1, 1]                                                     # Rank 3
    >>> dist.all_to_all_single(output, input, output_splits, input_splits)
    >>> output
    tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
    tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
    tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
    tensor([ 5, 17, 18, 24, 36])                                     # Rank 3
    '''

    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None, group=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output,
                                            output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits, group=ctx.group)
        return result, None, None, None
