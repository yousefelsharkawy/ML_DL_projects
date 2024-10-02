import math
from dataclasses import dataclass 
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
enc = tiktoken.get_encoding('gpt2')
import time
import inspect
import os
import numpy as np
from hellaswag import render_example, iterate_examples

# Modules ---------------------------------------------------------------------------
@dataclass 
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # vocab size of GPT-2 50,000 BPE merges + 256 single byte tokens + 1 special token <|endoftext|>
    n_layer: int = 12 # number of layers (how many times we repeat the block)
    n_head: int = 12 # number of heads in the multi-head attention
    n_embed: int = 768 # embedding dimension, so the head size is 768/12 = 64

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # convert the uint16 to int32 before casting to long, otherwise Pytorch doesn't like it
    ppt = torch.tensor(npt, dtype=torch.long) # torch.long is int64, which is what a lot of the layers uptop expect by default
    return ppt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {'train', 'val'}
        # get the shard filenames
        data_root = "/kaggle/input/fineweb-edu-10b/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s] # mine: filter the shards to only include the split we want, we will make 2 instances of DataLoaderLite, one for train and one for val
        shards = sorted(shards) # sort them so that they are in order
        shards = [os.path.join(data_root, s) for s in shards] # get the full path of the shards
        self.shards = shards
        assert len(shards) > 0, "no shards found for split {split}"
        if master_process:
            print(f"Found {len(shards)} shards for {split} split")
        self.reset()

    # put the reset code into a function to call it at the beginning of each validation step on the val loader
    def reset(self):
        # mine: load the tokens from the shards, we now have a starting position for each shard and whithin the shard
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank # the starting position for process_rank 0 is 0, for process_rank 1 is B*T, process_rank 2 is 2*B*T, etc (so they are now spaced out)

    # set the data loader to a specific state (from a checkpoint)
    def set(self, loader_checkpoint):
        self.current_position = loader_checkpoint['current_position'] + self.B * self.T * self.process_rank # we add the B*T*process_rank to the position to make sure it is the correct position for the process 
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        else:
            self.current_shard = loader_checkpoint['current_shard']
            self.tokens = load_tokens(self.shards[self.current_shard])
        

        
    def next_batch(self):
        B, T = self.B, self.T
        buffer = self.tokens[self.current_position:self.current_position + B*T + 1]
        x = buffer[:-1].view(B,T)
        y = buffer[1:].view(B,T)
        # advance the position
        self.current_position += B * T * self.num_processes # advance by the number of tokens in the batch * number of processes (the position has to advance by the entire chunk given to all processes)
        # if loading the next batch will exceed the tokens, reset the position and load the next shard
        if self.current_position + (B * T * self.num_processes + 1) >= len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
        dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # projects the n_embd features to vocab_size

        # parameter sharing between the embedding weights and the final linear layer
        self.transformer.wte.weight = self.lm_head.weight # we redirect the pointer of the embedding weights to the linear layer weights, the old embedding weights are orphaned, and python will garbage collect them

        # initialize the parameters, we call the apply method on self -which is a method implemented in nn.Module, it will iterate over all the submodules and apply the function to them-
        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # if has the NANOGPT_SCALE_INIT attribute, scale the std
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)**-0.5
            # initialize the weights of the linear layer with a normal distribution of mean 0 and std 0.02
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # if the linear layer has a bias, initialize it to zeros (by default pytorch initializes the bias to uniform)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self,x,y=None):
        # input x is the token sequence a tensor of shape (B,T) where B is the batch size and T is the sequence length
        B,T = x.size()
        assert T <= self.config.block_size, "Length of input tokens exceeds block size"
        ## get the token embeddings
        token_embeddings = self.transformer.wte(x) # shape (B,T,n_embed)
        
        ## get the positional encodings
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) # position indices shape (T)
        pos = self.transformer.wpe(pos) # convert them to embeddings, shape (T,n_embed)
        
        ## sum the token embeddings and positional embeddings
        x = token_embeddings + pos # shape (B,T,n_embed), the positional embeddings are broadcasted along the batch dimension
        
        ## forward through all the transformer blocks
        for block in self.transformer.h:
            x = block(x) # takes input of shape (B,T,n_embed) and returns the same shape
        # forward the final layer normalization and classifier
        x = self.transformer.ln_f(x) # shape (B,T,n_embed)
        logits = self.lm_head(x) # shape (B,T,vocab_size)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) # cross entropy loss
        
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """ Loads pretrained GPT-2 model weights from HuggingFace """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_type} model weights")
        
        ## Prepare the configuration
        # n_layer, n_head, and n_embed are determined from the model_type
        config_args = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embed=768), # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embed=1600) # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # the same for all GPT-2 models
        config_args['block_size'] = 1024 # the same for all GPT-2 models

        # initialize the model (our implementation)
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard the masks/buffers, not a parameter so we don't need to copy it

        # inita hugging face transformer model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        # get its state dict
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        #mine: these buffers are not in hugging face state dict anyway
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        # some of the weights in the hugging face model are transposed, so we need to transpose them back before copying them
        # this comes from the tensorflow repo
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        assert len(sd_keys_hf) == len(sd_keys), "Mismatched Keys {} != {}".format(len(sd_keys_hf), len(sd_keys))
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t()) # copy_ is an inplace copy, t() is the transpose
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model 

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # get all the parameters
        param_dict = {pn:p for pn,p in self.named_parameters()}
        # keep only those that require grad
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}

        # create optim groups, any parameters that are 2D will be weight_decayed, otherwise no
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't
        decay_params   = [p for n,p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay':  weight_decay},
            {'params': nodecay_params, 'weight_decay':  0.0         }
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # create Adamw optimizer and use the fused version if it is available (in later versions of Pytorch)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f'using Fused AdamW: {use_fused}')

        opt = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return opt
    

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp  = MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x)) # communication
        x = x + self.mlp(self.ln_2(x)) # computation, to think on what they gathered
        return x



class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0, 'n_embed should be divisible by n_head'
        # key, Query, and Value projections for all heads, but in a batch (mine: instead of separate matrices Key, Query, and Value)
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed) # we concatenated all key, query, and value in a single matrix (each one is n_embed which is further concatenation of n_head*head_size -so each is the concatenation of all heads-)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        # the mask, but we call it bias to match the huggingFace state Dict 
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size)) # reshape it to 4D tensor (1,1,block_size,block_size), so it will be reshaped later for all examples and heads
        
        self.n_head  = config.n_head
        self.n_embed = config.n_embed
        self.c_proj.NANOGPT_SCALE_INIT = 1

        
    def forward(self,x):
        B,T,C = x.size() # batch_size, sequence_length, n_embed
        qkv = self.c_attn(x) # batch_size, sequence_length, 3 * n_embed 
        q,k,v = qkv.split(self.n_embed,dim=2) # batch_size, sequence_length, n_embed for each (that is for all heads, each head will have part of that n_embed, precisely n_head = n_embed/n_head)
        # further split the q,k,v into multiple heads
        head_size = C // self.n_head # head_size = n_embed // number of heads
        k = k.view(B,T, self.n_head, head_size).transpose(1,2) # (batch_size, n_head, sequence_length, head_size), notice that we first reshaped the n_embed to n_head*head_size, then transposed
        q = q.view(B,T, self.n_head, head_size).transpose(1,2) # (batch_size, n_head, sequence_length, head_size)
        v = v.view(B,T, self.n_head, head_size).transpose(1,2) # (batch_size, n_head, sequence_length, head_size)

        # # compute the attention scores (affinities) for each example and each head
        # wei = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # (batch_size, n_head, sequence_length, sequence_length), then divide by sqrt(head_size) to normalize
        # # discard the future tokens for each token
        # wei = wei.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # mask the future tokens
        # # apply the softmax to get the attention weights
        # wei = F.softmax(wei, dim=-1)
        # # use the attention weights to get the weighted sum of the values
        # y = wei @ v # (batch_size, n_head, sequence_length, sequence_length) @ (batch_size, n_head, sequence_length, head_size) = (batch_size, n_head, sequence_length, head_size)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # concatenate the heads together 
        y = y.transpose(1,2).contiguous().view(B,T,C) # first transpose to (batch_size, sequence_length, n_head, head_size) then contiguous to make sure the memory is contiguous, then view to (batch_size, sequence_length, n_embed = n_head * head_size)

        # output projection
        y = self.c_proj(y)
        return y
    
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def save_checkpoint(model,optimizer,scalar,step,train_loader, val_loss, checkpoint_path='gpt2_checkpoint.pth'):
    # data loader checkpoint
    train_loader_checkpoint = {'current_shard': train_loader.current_shard, 'current_position': train_loader.current_position}
    checkpoint = {
        'train_loader': train_loader_checkpoint,
        'model': model.state_dict(),
        'config': model.config,
        'optimizer': optimizer.state_dict(),
        'scaler': scalar.state_dict(),
        'step': step,
        'val_loss':val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path} at step {step} and val_loss {val_loss}")



## setup the DDP (Distributed Data Parallel) ----------------------------------------------
# Simple Launch: python train_gpt2.py
# DDP Launch (for n GPUs): torchrun --standalone --nproc_per_node=n train_gpt2.py
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to the rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    # get the local rank, global rank, and world size from the environment
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    # set the device to the local rank
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # we will choose on of them to be the master process, which will do logging, checkpointing, etc (other processes will be just for computation)
else:
    # Vanilla, non-DDP training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")


## initializations ----------------------------------------------------------------

### set the seed for reproducibility
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

### calculate the gradient accumulation steps (to simulate a larger batch size)
total_batch_size = 524288 # 2**19, ~0.5M tokens desired total tokens 
B = 8 # micro batch size, mine: set to 8 to fit in memory
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "Make sure total_batch_size is divisible by B*T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size) # number of steps to accumulate the gradients over
# print using only the master process
if master_process:
    print(f"Total desired batch size: {total_batch_size:,}")
    print(f"Number of gradient accumulation steps: {grad_accum_steps}")

### initialize the data loader
train_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='train')
val_loader = DataLoaderLite(B=B, T=T, process_rank = ddp_rank, num_processes = ddp_world_size, split='val')


### Initialize the model
# use the hugging face model
# model = GPT.from_pretrained('gpt2')
# initialize the model from scratch
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)

### initialize the optimizer
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

### initialize the gradient scaling for mixed precision training
scaler = torch.amp.GradScaler(device=device)

### initial the step we are currently at 
current_step = 0

### create the log directory for checkpoints and logs (train loss, validation loss, hellaswag accuracies)
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

resume_training = False
if resume_training:
    checkpoint_path = '/kaggle/input/gpt2-124m/gpt2_checkpoint.pth'
    log_file = 'log/log.txt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_loader.set(checkpoint['train_loader'])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    current_step = checkpoint['step'] + 1 # we will start from the next step
    batch_val_loss = torch.tensor(checkpoint['val_loss'])
    if master_process:
        print(f"Resuming training from step {current_step} with validation loss {batch_val_loss.item():.4f}")
else:
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, 'w') as f:  # open for writing to clear the file, then we will append to it below
        pass 
    

max_lr = 6e-4
min_lr = max_lr  * 0.1
# in the GPT paper they said they warmed up the lr over 375M tokens, so the warmup steps are 375M / 2**19 = 715.25 (100 was good enough though)
warmup_steps = 715
# total 10B tokens / 2**19 token per step = 10*10**9 / 2**19 = 19073.48633 steps
max_steps = 19073 # after it we will continue at 10% of the original learning rate (min_lr)
def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * ((it + 1) / warmup_steps) # the +1 so that we don't really start with lr=0 (won't update the weights)
    
    # 2) if we passed the learning rate decay iterations, we continue at 10% of the original learning rate (min_lr)
    if it > max_steps:
        return min_lr
    
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps) # the decay ratio will start from 0 and reach 1 at max_steps
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # cosine decay, coeff starts at 1 and reaches 0 at max_steps
    return min_lr + (max_lr - min_lr) * coeff





# torch compile the model
use_compile = False # torch compile interfers with the generation code TODO fix
if use_compile:
    model = torch.compile(model)
# if ddp, wrap the model in DDP container (it will handle the communication between the processes' models)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # contains the "raw" unwrapped model in case we want to save it or do something else with it (mine: instead of calling model.module everywhere)



## Training loop ----------------------------------------------------------------
for step in range(current_step,max_steps):
    t0 = time.time()

    last_step = (step == max_steps - 1) # we want to calculate the val loss and generate on the last step of the training 
    
    ### once in a while, evaluate the validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            # accumulate the gradients over say, 20 steps (mine in each step we process B * T * ddp_world_size: so B * T * ddp_world_size * val_loss_steps tokens)
            batch_val_loss = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                batch_val_loss += loss.detach()
            if ddp:
                dist.all_reduce(batch_val_loss, op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {batch_val_loss.item():.4f}")
                with open(log_file, 'a') as f:
                    f.write(f"{step} val {batch_val_loss.item():.4f}\n")
    
    ### once in a while, evaluate the hellaswag loss
    if (step % 500 == 0 or last_step) and (not use_compile):
        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes (sum the total and correct predictions for them)
        if ddp:
            # package the statistics into tensors (to call all_reduce on them)
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            # sum the total examples and total correct examples across all processes
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")


    ### once in a while, generate from the model (except for step 0)
    if ((step > 0 and step % 1000 == 0) or last_step) and (not use_compile):
        ### Generation
        model.eval()
        num_return_sequences = 4
        max_length = 32
        # get the prefix tokens
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # shape (T)
        x = tokens.unsqueeze(0).repeat(num_return_sequences, 1).to(device)  # shape (num_return_sequences, T)
        # we created a genrator object in pytorch sepciically for the sampling
        # we don't want to affect the global random state that is used for training
        sample_rng = torch.Generator(device=device) # create a generator for the sampling
        # we seed it differently for every rank, and we will make them all print their generations
        sample_rng.manual_seed(42 + ddp_rank)
        while x.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x)
                # take the logits at the last position, we only care about the last token's logits
                logits = logits[:, -1, :] # shape (num_return_sequences, vocab_size)
                # get the probabilities by applying softmax
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of k = 50 (in which we get the top k tokens and sample from them)
                # this is hugging face's pipeline default
                # topk_probs and topk_indices are of shape (num_return_sequences, 50)
                topk_props, topk_indices = torch.topk(probs, 50, dim=-1) # get the top 50 tokens and their probabilities
                # sample a token from the top 50 tokens
                ix = torch.multinomial(topk_props, num_samples=1, generator=sample_rng) # the indices of chosen tokens (in range 0-49)
                # use the indices to index to the actual indices (get the actual tokens)
                next_token = torch.gather(topk_indices, -1, ix) # use the indices to index to the actual indices 
                # append the next token to the sequence
                x = torch.cat((x, next_token), dim=1)

        # decode the tokens
        for i in range(num_return_sequences):
            tokens = x[i].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    ### do one step of optimization
    model.train()
    # mine: zero the gradients before the next big batch (more precisely before the grad_accum_steps on small batches)
    optimizer.zero_grad()
    batch_loss = 0 # just to print it
    # for grad accumulation steps, do the forward then the backward multiple times and accumulate the gradients
    for micro_step in range(grad_accum_steps):
        ## get the next batch
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)

        # Confusingly, `model.require_backward_grad_sync` is actually used by both the forward and backward pass. Moved up the line so that it also gets applied to the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) # will be true only for the last step of the accumulation loop
        
        ## forward prop and Loss calculation
        # # for mixed precision training (float16)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x,y)
        #logits, loss = model(x,y)
        loss /= grad_accum_steps # divide the loss by the number of accumulation steps
        
        ## backward prop (scale the loss before backprop to avoid underflowing gradients)
        scaler.scale(loss).backward()
        #loss.backward()

        # to print the batch loss later
        batch_loss += loss.detach() # detach so that we detach the batch_loss from the computation graph


    # average the batch loss as well over all the ranks (it is outside the DPP container)
    if ddp:
        dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)

    ## gradient clipping
    # unscale the gradients before clipping (to get back to the original scale)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # this will unscale the gradients unless it is called above (which we did) then updates the weights if the gradients have no inf or nan
    scaler.step(optimizer)
    # this will update the scaler for the next iteration
    scaler.update()
    torch.cuda.synchronize()
    t1 = time.time()
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1-t0)
    if master_process:
        print(f"step {step:4d} | loss: {batch_loss.item():.6f} | lr: {lr:.4e}  | norm: {norm:.4f} | dt: {(t1-t0) * 1000} ms, toks/sec: {tokens_per_sec}")
        with open(log_file, 'a') as f:
            f.write(f"{step} train {batch_loss.item():.6f}\n")
        if step % 50 == 0 or last_step:
            save_checkpoint(raw_model, optimizer, scaler, step, train_loader, batch_val_loss.item())

    # if we finished 2500 steps, break from the loop (for kaggle time limit)
    if step % 2500 == 0 and step > 0:
        break


# cleanup (call destroy the process group)
if ddp:
    destroy_process_group()
    

