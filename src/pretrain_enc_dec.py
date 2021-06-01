# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False

from datetime import datetime
import os
import random
import math
import numpy as np
import torch
import json
from tqdm import tqdm

import deepspeed

from arguments import get_args
from tokenization_enc_dec import EncDecTokenizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
from learning_rates import AnnealingLR
from model import EncDecModel, EncDecConfig
from model import enc_dec_get_params_for_weight_decay_optimization

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP
import mpu
from apex.optimizers import FusedAdam as Adam
from utils import Timers
from utils import save_checkpoint
from utils import load_checkpoint
from utils import report_memory
from utils import print_args
from utils import print_rank_0, save_rank_0
import torch.distributed as dist

from data.enc_dec_dataset import build_train_valid_test_datasets
from samplers import DistributedBatchSampler


def get_model(args, vocab_size):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    model = EncDecModel(config,
                        parallel_output=True,
                        checkpoint_activations=args.checkpoint_activations,
                        checkpoint_num_layers=args.checkpoint_num_layers)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = enc_dec_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)

    return lr_scheduler


def setup_model_and_optimizer(args, vocab_size):
    """Setup model and optimizer."""

    model = get_model(args, vocab_size)
    optimizer = get_optimizer(model, args)
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )

    print(args.load)
    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def get_masks_and_position_ids(args,
                               tokenizer: EncDecTokenizer,
                               contexts,
                               targets,
                               labels,
                               ctx_eod_mask,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.ones(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    # Enc Position ids.
    enc_pos_ids = torch.arange(
        enc_seq_length, dtype=torch.long, device=contexts.device)
    enc_pos_ids = enc_pos_ids.unsqueeze(0).expand_as(contexts)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        enc_pos_ids = enc_pos_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            eod_pos = ctx_eod_mask[b].nonzero(as_tuple=False)
            prev_index = 0
            for i in eod_pos:
                if i < enc_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                            enc_attn_mask[b, 0, i+1:, :i+1] = 0
                            enc_attn_mask[b, 0, :i+1, i+1:] = 0
                    # Reset positions.
                    if reset_position_ids:
                        enc_pos_ids[b, i+1:] -= (i + 1 - prev_index)
                        prev_index = i + 1

    
    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

    # Dec Position ids.
    dec_pos_ids = torch.arange(
        dec_seq_length, dtype=torch.long, device=targets.device)
    dec_pos_ids = dec_pos_ids.unsqueeze(0).expand_as(targets)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        dec_pos_ids = dec_pos_ids.clone()

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            eod_pos = (targets[b] == tokenizer.eod_id).nonzero(as_tuple=False)
            prev_index = 0
            for i in eod_pos:
                if i < dec_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                        dec_attn_mask[b, 0, i+1:, :i+1] = 0
                        dec_attn_mask[b, 0, :i+1, i+1:] = 0
                    # Reset positions.
                    if reset_position_ids:
                        dec_pos_ids[b, i+1:] -= (i + 1 - prev_index)
                        prev_index = i + 1

    # Loss mask.
    loss_mask = torch.ones(targets.size(), dtype=torch.float, device=targets.device)
    loss_mask[targets == tokenizer.eod_id] = 0.0
    loss_mask[labels == tokenizer.pad_id] = 0.0

    # Cross Attention Mask
    cross_attn_mask = torch.ones(
        batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)

    if reset_position_ids or reset_attention_mask:
        for b in range(batch_size):
            enc_eod_pos = ctx_eod_mask[b].nonzero(as_tuple=False)
            dec_eod_pos = (targets[b] == tokenizer.eod_id).nonzero(as_tuple=False)
            assert len(enc_eod_pos) == len(dec_eod_pos), (enc_eod_pos, dec_eod_pos)
            for enc_i, dec_i in zip(enc_eod_pos, dec_eod_pos):
                if enc_i < enc_seq_length and dec_i < dec_seq_length:
                    # reset attentions
                    if reset_attention_mask:
                        cross_attn_mask[b, 0, dec_i+1:, :enc_i+1] = 0
                        cross_attn_mask[b, 0, :dec_i+1, enc_i+1:] = 0

    if args.fp16:
        enc_attn_mask = enc_attn_mask.half()
        dec_attn_mask = dec_attn_mask.half()
        cross_attn_mask = cross_attn_mask.half()

    model_batch = {
        "enc_attention_mask": enc_attn_mask,
        "enc_position_ids": enc_pos_ids,
        "dec_attention_mask": dec_attn_mask,
        "dec_position_ids": dec_pos_ids,
        "cross_attention_mask": cross_attn_mask,
    }

    no_model_batch = {
        "loss_mask": loss_mask
    }

    return model_batch, no_model_batch


def get_batch(tokenizer, data_iterator, args, timers):
    ''' get_batch subdivides the source data into chunks of
    length args.seq_length. If source is equal to the example
    output of the data loading example, with a seq_length limit
    of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the data loader. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. A Variable representing an appropriate
    shard reset mask of the same dimensions is also returned.
    '''
    # Items and their type.
    datatype = torch.int64

    # Broadcast data.
    # timers('data loader').start()

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    keys = [
        "contexts",
        "targets",
        "labels",
        "ctx_eod_mask",
    ]

    # timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    contexts = data_b['contexts'].long()
    targets = data_b['targets'].long()
    labels = data_b['labels'].long()
    ctx_eod_mask = data_b['ctx_eod_mask'].long()

    # Get the masks and postition ids.
    model_b, no_model_b = get_masks_and_position_ids(
        args,
        tokenizer,
        contexts,
        targets,
        labels,
        ctx_eod_mask,
        args.reset_position_ids,
        args.reset_attention_mask)

    batch = {
        "enc_input_ids": contexts,
        "dec_input_ids": targets,
        **model_b
    }

    no_model_batch = {
        "labels": labels,
        **no_model_b
    }

    return batch, no_model_batch


def forward_step(tokenizer, data_iterator, model, args, timers):
    """Forward step."""

    # Get the batch.
    # timers('batch generator').start()
    # tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator, args, timers)
    batch, no_model_batch = get_batch(tokenizer, data_iterator, args, timers)
    # timers('batch generator').stop()
        
    # Forward model.
    output = model(**batch)
    logits = output["lm_logits"]
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])
    loss_mask = no_model_batch["loss_mask"].view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    return loss


def backward_step(optimizer, model, lm_loss, args, timers):
    """Backward step."""
    # Total loss.

    loss = lm_loss

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Reduce across processes.
    lm_loss_reduced = lm_loss

    reduced_losses = lm_loss.view(1)

    if args.deepspeed:
        # DeepSpeed backward propagation already addressed all reduce communication.
        # Reset the timer to avoid breaking timer logs below.
        timers('allreduce').reset()
    else:
        torch.distributed.all_reduce(reduced_losses.data)
        reduced_losses.data = reduced_losses.data / args.world_size
        if not USE_TORCH_DDP:
            timers('allreduce').start()
            model.allreduce_params(reduce_after=False,
                                   fp32_allreduce=args.fp32_allreduce)
            timers('allreduce').stop()

    lm_loss_reduced = reduced_losses

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)

    return lm_loss_reduced


def see_memory_usage(message, force=False):
    if not force:
        return
    dist.barrier()
    if dist.get_rank() == 0:
        print(message)
        print("Memory Allocated ", torch.cuda.memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Max Memory Allocated ", torch.cuda.max_memory_allocated()/(1024*1024*1024), "GigaBytes")
        print("Cache Allocated ", torch.cuda.memory_cached()/(1024*1024*1024), "GigaBytes")
        print("Max cache Allocated ", torch.cuda.max_memory_cached()/(1024*1024*1024), "GigaBytes")
        print(" ")
        # input("Press Any Key To Continue ..")


def train_step(tokenizer, data_iterator, model, optimizer, lr_scheduler,
               args, timers):
    """Single training step."""

    lm_loss = forward_step(tokenizer, data_iterator, model, args, timers)
    lm_loss_reduced = backward_step(optimizer, model, lm_loss, args, timers)

    if dist.get_rank() == 0:
        print("loss", lm_loss_reduced)

    # Update parameters.
    skipped_iter = 0
    if args.deepspeed:
        model.step()
    else:
        optimizer.step()

        # Update learning rate.
        if not (args.fp16 and optimizer.overflow):
            lr_scheduler.step()
        else:
            skipped_iter = 1

    return lm_loss_reduced, skipped_iter


def train(tokenizer, model, optimizer, lr_scheduler,
          train_data_iterator, val_data_iterator, timers, args):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    for iteration in tqdm(range(args.iteration, args.train_iters), disable=(torch.distributed.get_rank() != 0), desc="Pretaining"):

        lm_loss, skipped_iter = train_step(tokenizer, train_data_iterator,
                                           model,
                                           optimizer,
                                           lr_scheduler,
                                           args, timers)
        skipped_iters += skipped_iter

        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()

        # Logging.
        if iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            elapsed_time = timers('interval time').elapsed()
            log_string = ' iteration {:8d}/{:8d} |'.format(iteration,
                                                            args.train_iters)
            log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
                elapsed_time * 1000.0 / args.log_interval)
            log_string += ' learning rate {:.3} |'.format(learning_rate)
            log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
            if args.fp16:
                log_string += ' loss scale {:.1f} |'.format(
                    optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
            print_rank_0(log_string)
            save_rank_0(args, log_string)
            total_lm_loss = 0.0
            if report_memory_flag:
                report_memory('after {} iterations'.format(iteration))
                report_memory_flag = False
        # Checkpointing
        if args.save and args.save_interval and iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and args.do_valid:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(
                tokenizer, prefix, val_data_iterator, model, args, timers, False)

        if args.exit_interval and iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, iteration), flush=True)
            exit()

    return iteration, skipped_iters


def evaluate(tokenizer, data_iterator, model, args, timers, verbose=False):
    """Evaluation."""

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_lm_loss = 0

    with torch.no_grad():
        for iteration in tqdm(range(args.eval_iters), disable=(torch.distributed.get_rank() != 0), desc="Evaluating"):
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                msg = 'Evaluating iter {}/{}'.format(iteration, args.eval_iters)
                print_rank_0(msg)
                save_rank_0(args, msg)
            # Forward evaluation.
            lm_loss = forward_step(tokenizer, data_iterator, model, args, timers)

            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()

            # Reduce across processes.
            if isinstance(model, DDP):
                torch.distributed.all_reduce(lm_loss.data)
                lm_loss.data = lm_loss.data / args.world_size

            total_lm_loss += lm_loss.data.detach().float().item()

    # Move model back to the train mode.
    model.train()

    total_lm_loss /= args.eval_iters
    return total_lm_loss


def evaluate_and_print_results(tokenizer, prefix, data_iterator, model,
                               args, timers, verbose=False):
    """Helper function to evaluate and dump results on screen."""
    lm_loss = evaluate(tokenizer, data_iterator, model, args, timers, verbose)
    lm_ppl = math.exp(min(20, lm_loss))
    string = '-' * 100 + "\n"
    string += ' validation loss at {} | '.format(prefix)
    string += 'LM loss: {:.6} | '.format(lm_loss)
    string += 'LM PPL: {:.6}'.format(lm_ppl)
    length = len(string) + 1
    string = '-' * length + "\n" + string + "\n" + '-' * length
    print_rank_0(string)
    save_rank_0(args, string)

    return lm_loss


def set_deepspeed_activation_checkpointing(args):

    deepspeed.checkpointing.configure(mpu, deepspeed_config=args.deepspeed_config, num_checkpoints=args.num_layers)
    mpu.checkpoint = deepspeed.checkpointing.checkpoint
    mpu.get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
    mpu.model_parallel_cuda_manual_seed = deepspeed.checkpointing.model_parallel_cuda_manual_seed


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    init_method = 'tcp://'
    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method += master_ip + ':' + master_port
    deepspeed.init_distributed()

    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)

    # Optional DeepSpeed Activation Checkpointing Features
    #
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def make_data_loader(dataset):
    """Buld dataloader given an input dataset."""
    if dataset is None:
        return None
    args = get_args()

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    # Use a simple sampler with distributed batch sampler.
    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True)


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)

    # Pytorch distributed.
    initialize_distributed(args)
    if torch.distributed.get_rank() == 0:
        print('Pretrain Enc-Dec model')
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # setup tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    
    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    args.gradient_accumulation_steps = ds_config["gradient_accumulation_steps"]

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, tokenizer.vocab_size)
    optimizer.cur_scale = 4096
    
    if torch.distributed.get_rank() == 0:
        print(args.iteration)
    
    train_data_iterator, val_data_iterator, test_data_iterator = \
            build_train_valid_test_data_iterators(
                    train_valid_test_dataset_provider, args, tokenizer)

    iteration = 0
    if args.train_iters > 0:
        iteration, skipped = train(tokenizer, model, optimizer,
                                   lr_scheduler,
                                   train_data_iterator,
                                   val_data_iterator,
                                   timers, args)

        prefix = 'the end of training for val data'
        evaluate_and_print_results(tokenizer, prefix, val_data_iterator,
                                                  model, args, timers, False)

    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    if args.do_test:
        # Run on test data.
        prefix = 'the end of training for test data'
        evaluate_and_print_results(tokenizer, prefix, test_data_iterator,
                                   model, args, timers, True)


def train_valid_test_dataset_provider(tokenizer, train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for Enc-Dec ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        tokenizer=tokenizer,
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        enc_seq_length=args.enc_seq_length,
        dec_seq_length=args.dec_seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating Enc-Dec datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_data_iterators(
        build_train_valid_test_datasets_provider, args, tokenizer):
    """XXX"""

    (train_dataloader, valid_dataloader, test_dataloader) = (None, None, None)

    print_rank_0('> building train, validation, and test datasets ...')
    # Data loader only on rank 0 of each model parallel group.
    if mpu.get_model_parallel_rank() == 0:
        # Rank, size, and global batch size.
        data_parallel_size = mpu.get_data_parallel_world_size()
        global_batch_size = args.batch_size * data_parallel_size

        # Number of train/valid/test samples.
        train_iters = args.train_iters
        eval_iters = (train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_iters * global_batch_size,
                                      eval_iters * global_batch_size,
                                      test_iters * global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
        print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
        print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))

        # Build the datasets.
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets_provider(
            tokenizer, train_val_test_num_samples)

        # Build dataloders.
        train_dataloader = make_data_loader(train_ds)
        valid_dataloader = make_data_loader(valid_ds)
        test_dataloader = make_data_loader(test_ds)

        # Flags to know if we need to do training/validation/testing.
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        # Need to broadcast num_tokens and num_type_tokens.
        flags = torch.cuda.LongTensor(
            [int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])

    # Broadcast num tokens.
    torch.distributed.broadcast(flags,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()

    # Shift the start iterations.
    if train_dataloader is not None:
        train_dataloader.batch_sampler.start_iter = args.iteration % \
            len(train_dataloader)
        print_rank_0('setting training data start iteration to {}'.
                     format(train_dataloader.batch_sampler.start_iter))
    if valid_dataloader is not None:
        start_iter_val = (args.iteration // args.eval_interval) * \
            args.eval_iters
        valid_dataloader.batch_sampler.start_iter = start_iter_val % \
            len(valid_dataloader)
        print_rank_0('setting validation data start iteration to {}'.
                     format(valid_dataloader.batch_sampler.start_iter))

    # Build iterators.
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader)
    else:
        train_data_iterator = None

    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader)
    else:
        valid_data_iterator = None

    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


if __name__ == "__main__":
    main()
