"""FastMix training with the *validation split of the training data* as the search target.

This script jointly trains a small proxy language model and searches for the optimal
data-mixture weights (``dataset_probs``) across training sources. The mixture is optimized
so that a one-step model update on the weighted training batch minimizes the loss on a
held-out validation split of the training data (see ``--val_data_dir``).

Counterpart: ``train_fastmix_sft.py`` uses downstream SFT data as the search target instead.
"""
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import copy
import lightning as L
import torch
import torch.nn as nn
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM
import torch.nn.functional as F
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.model import GPT, Block, Config, CausalSelfAttention
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from lit_gpt import FusedCrossEntropyLoss
import random
import yaml
import os
import wandb

# Log in to Weights & Biases using the WANDB_API_KEY environment variable.
# Set `export WANDB_MODE=disabled` to disable logging entirely.
if os.environ.get("WANDB_API_KEY"):
    wandb.login(key=os.environ["WANDB_API_KEY"])

# tinyllama_1M, tinyllama_60M, tinyllama_1_1b
model_name = "tinyllama_1_1b"

# Experimental settings
reset_embedding = False
group_level_sampling = False
only_save_model = False

# Hyperparameters
total_devices = 8
num_of_devices = 8
num_of_nodes = total_devices // num_of_devices if total_devices >= num_of_devices else 1
# optimal tokens should be 10^20
global_batch_size = 512
learning_rate = 4e-4
min_lr = 1e-5
decay_lr = True

micro_batch_size = 4
max_step = 25000
warmup_steps = 1000
log_step_interval = 10
eval_iters = 50
save_step_interval = 1000
eval_step_interval = 1000
# -100 is the default ignore index
# ignore_token_id = -100

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

batch_size = global_batch_size // total_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps


max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# Be careful about the weights, it should be something as the len(dataset) * actual reweighting
train_data_config = [
]
val_data_config = [
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# get a random name
random_name = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=10))
logger = step_csv_logger("out", random_name, flush_logs_every_n_steps=log_iter_interval)
# log hyper-parameters into wandb
#wandb_logger = WandbLogger()

def setup(
    data_seed: int = 3406,
    devices: int = num_of_devices,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    data_yaml_file: Optional[Path] = None,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = False,
    out_name: str = "default_model",
    load_from: Optional[Path] = None,
    gpu_memory: Optional[int] = None, 
    learning_rate_dataset: float = 0.0001,
    #gpu_ids: str = "0",
) -> None:
    # may modify the global train_data_config
    global train_data_config
    global val_data_config
    wandb_logger = WandbLogger(name=out_name)
    
    # if train config exists as a yaml file, load it
    if data_yaml_file is not None:
        data_yaml_file = Path(data_yaml_file)
        if data_yaml_file.exists():
            print("loading config from {}".format(data_yaml_file))
            with open(data_yaml_file, "r") as f:
                # template yaml file is as
                # train_file: weight
                config = yaml.safe_load(f)
            if "data_seed" in config:
                data_seed = int(config["data_seed"])
                print("update data_seed to {}".format(data_seed))
            if "train" in config:
                train_config = []
                for k, v in config["train"].items():
                    train_config.append((k, float(v)))
                # update the config
                train_data_config = train_config
            if "valid" in config:
                val_config = []
                for k, v in config["valid"].items():
                    # TODO: by deafult we use separate validation set
                    val_config.append([(k, float(v))])
                val_data_config = val_config
                    # 093 train and valid
            del config["train"]
            del config["valid"]
            # see if any local variable is in the config, if so, update it
            for k, v in config.items():
                if k in globals():
                    print("update {} to {}".format(k, v))
                    globals()[k] = v
            # use the new value to update
            if "num_of_devices" in config.keys():
                # update devices
                devices = num_of_devices
            # if global batch size is changed, update the batch size
            if "global_batch_size" in config.keys():
                globals()["batch_size"] = global_batch_size // total_devices
                globals()["gradient_accumulation_steps"] = batch_size // micro_batch_size
                globals()["warmup_iters"] = warmup_steps * gradient_accumulation_steps
                globals()["max_iters"] = max_step * gradient_accumulation_steps
                globals()["lr_decay_iters"] = max_iters
                globals()["log_iter_interval"] = log_step_interval * gradient_accumulation_steps                
        else:
            print("config {} does not exist, skip loading".format(data_yaml_file))
    
    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            strategy = FSDPStrategy(
                # nn.Embedding
                auto_wrap_policy={Block,nn.Embedding},
                # activation_checkpointing_policy={Block},
                state_dict_type="full",
                sharding_strategy="FULL_SHARD",
                limit_all_gathers=True,
                cpu_offload=False,
            )
    else:
        strategy = "auto"
        
    fabric = L.Fabric(devices=devices, 
                      strategy=strategy, 
                      precision=precision, 
                      loggers=[logger, wandb_logger],
                      num_nodes=num_of_nodes)

    fabric.print("precision: {}".format(precision))
    fabric.print("Use gpu memory: {}".format(gpu_memory))

    hparams = {k: v for k, v in globals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
    fabric.print(hparams)
    wandb_logger.log_hyperparams(hparams)
    # log the train & val data config
    wandb_logger.log_hyperparams({"train_data_config": train_data_config, "val_data_config": val_data_config})
    fabric.print(train_data_config)
    #fabric.launch(main, train_data_dir, val_data_dir, resume)
    main(fabric, data_seed, train_data_dir, val_data_dir, resume, out_name, load_from, learning_rate_dataset)


def main(fabric, data_seed, train_data_dir, val_data_dir, resume, out_name, load_from, learning_rate_dataset):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)
    out_dir = Path("checkpoints") / out_name

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = LlamaConfig(
        vocab_size=50432,
        hidden_size=256,              # n_embd
        intermediate_size=512,        # intermediate_size
        num_hidden_layers=2,          # n_layer
        num_attention_heads=8,        # n_head
        max_position_embeddings=2048, # block_size
        rms_norm_eps=1e-5,            # norm_eps
    )
    model = LlamaForCausalLM(config)

    train_dataloader, val_dataloaders = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=model.config.max_position_embeddings,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=data_seed,
    )
    
    if load_from is not None:
        print("loading model from {}".format(load_from))
        state_dict = torch.load(load_from, map_location=fabric.device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)  # strict=False 兼容部分参数

    fabric.print(f"Time to instantiate model: {time.perf_counter() - 0:.02f} seconds.")
    fabric.print(f"Total parameters {model.num_parameters():,}")

    model = fabric.setup(model)

    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if val_dataloaders is not None:
        for i in range(len(val_dataloaders)):
            val_dataloaders[i] = fabric.setup_dataloaders(val_dataloaders[i])

    fabric.seed_everything(data_seed)  # same seed for every process to init model (FSDP) 
    t0 = time.perf_counter()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), fused=True
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)
    
    class ProbsModule(nn.Module):
        def __init__(self, probs):
            super().__init__()
            self.probs = probs
    
    K = len(train_data_config)
    dataset_probs = torch.nn.Parameter(torch.ones(K, requires_grad=True))
    probs_module = ProbsModule(dataset_probs)
    probs_module = fabric.setup_module(probs_module)
    #dataset_probs = probs_module.probs

    # 为参数向量创建优化器，设置 L2 正则化
    #learning_rate_dataset = 0.0001
    dataset_optimizer = torch.optim.AdamW([probs_module.probs], lr=learning_rate_dataset, weight_decay=0, betas=(beta1, beta2), fused=True)
    #learning_rate_dataset = 1.0
    #dataset_optimizer = torch.optim.SGD([probs_module.probs], lr=learning_rate_dataset, momentum=0.9, weight_decay=0)
    dataset_optimizer = fabric.setup_optimizers(dataset_optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0, "dataset_probs": probs_module, "dataset_optimizer": dataset_optimizer}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))
        
    if resume:
        # take the last checkpoint
        resume = resume[-1]
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloaders, monitor, resume, out_dir)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

def delete_all_except_last(folder_path, file_extension=".pth"):
    # 获取文件夹中以特定扩展名结尾的所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
    if len(files) >= 2:
        files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
        
        for file_name in files[:-1]:
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    else:
        print("Not enough files to delete.")


def train(fabric, state, train_dataloader, val_dataloaders, monitor, resume, out_dir):
    model = state["model"]
    optimizer = state["optimizer"]
    probs_module = state["dataset_probs"]
    dataset_optimizer = state["dataset_optimizer"]

    if val_dataloaders is not None:
        # for i in range(len(val_dataloaders)):
        #     validate(fabric, model, val_dataloaders[i])  # sanity check
        # validate(fabric, model, val_dataloaders[0])
        pass

    # with torch.device("meta"):
    #     meta_model = GPT(model.config)
    #     # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
    #     # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
    #     # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
    #     estimated_flops = estimate_flops(meta_model) * micro_batch_size
    #     fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
    #     x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
    #     # measured_flos run in meta. Will trigger fusedRMSNorm error
    #     #measured_flops = measure_flops(meta_model, x)
    #     #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
    #     del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    if fabric.device.type == "xla":
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    
    
    initial_iter = state["iter_num"]
    curr_iter = 0
            
    loss_func = FusedCrossEntropyLoss()
    num_epochs = 3  # 你想要的轮数
    for epoch in range(num_epochs):
        for  train_data, group_ids in train_dataloader:
            # resume loader state. This is not elegant but it works. Should rewrite it in the future.
            # if resume:
            #     if curr_iter < initial_iter:
            #         curr_iter += 1
            #         fabric.print("skip iter {}".format(curr_iter))
            #         continue
            #     else:
            #         resume = False
            #         curr_iter = -1
            #         fabric.barrier()
            #         fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
            # if state["iter_num"] >= max_iters:
            #     break  
            
            # determine and set the learning rate for this iteration
            lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()

            input_ids = train_data[:, 0 : model.config.max_position_embeddings].contiguous()
            targets = train_data[:, 1 : model.config.max_position_embeddings + 1].contiguous()
            
            if state["iter_num"]!=0 and state["iter_num"] % eval_step_interval == 0 and val_dataloaders is not None:
                
                # 步骤 1: 计算 validation loss，并计算模型 head 针对该 loss 的梯度
                model.eval()
                val_loss = 0
                val_head_grads = []
                for i in range(len(val_dataloaders)):
                    for val_data, _ in val_dataloaders[i]:
                        #fabric.print("Check \n\n\n")
                        #fabric.print(val_data.shape, model.config.block_size)
                        input_ids_val = val_data[:, 0 : model.config.max_position_embeddings].contiguous()
                        targets_val = val_data[:, 1 : model.config.max_position_embeddings + 1].contiguous()
                        with torch.enable_grad():
                            logits_val = model(input_ids_val).logits
                            loss = chunked_cross_entropy(logits_val, targets_val, chunk_size=0)
                            optimizer.zero_grad()
                            fabric.backward(loss)
                            val_head_grads.append([p.grad.detach().clone() for p in model.lm_head.parameters()])
                        break   
                if val_head_grads:
                    # 逐参数平均
                    val_head_grad = []
                    for params in zip(*val_head_grads):
                        avg_param = sum(param for param in params) / len(params)
                        val_head_grad.append(avg_param)
                else:
                    val_head_grad = None
                model.train()
                # 步骤 2: 采样一个 batch 的训练数据，计算每个 group 的样本对应的模型 head 的梯度
                group_num = len(train_data_config)
                group_gradients = [[] for _ in range(group_num)]

                for group_id in range(group_num):
                    group_mask = group_ids == group_id
                    if group_mask.any():
                        group_input_ids = input_ids[group_mask]
                        group_targets = targets[group_mask]
                        with torch.enable_grad():
                            logits = model(group_input_ids).logits
                            loss = chunked_cross_entropy(logits, group_targets, chunk_size=0) #loss_func(logits, group_targets)
                        optimizer.zero_grad()
                        fabric.backward(loss)
                        group_head_grad = [p.grad.detach().clone() for p in model.lm_head.parameters()]
                        group_gradients[group_id] = group_head_grad
                optimizer.zero_grad()

                # 步骤 3: 计算步骤 1 和步骤 2 的梯度内积
                inner_products = []
                for group_grad in group_gradients:
                    if group_grad:
                        inner_prod = sum([(val_grad * group_g).sum() for val_grad, group_g in zip(val_head_grad, group_grad)])
                        inner_products.append(inner_prod.item())
                    else:
                        inner_products.append(0.0)

                fabric.print(f"Inner products for each group: {inner_products}")
                
                def compute_dataset_probs_gradient(dataset_probs, inner_products, learning_rate, lambda_reg):
                    # 计算 softmax 后的概率分布
                    p = F.softmax(dataset_probs, dim=0)
                    # 计算梯度内积乘以学习率部分 (注意符号)
                    g = - learning_rate * torch.tensor(inner_products, device=dataset_probs.device)
                    dgdp = g  # 假设 g 对 p 的导数就是 g 本身，具体根据实际情况调整
                    # 计算梯度内积部分关于 dataset_probs 的导数
                    dgdp_expanded = dgdp.unsqueeze(1).expand(-1, p.size(0))
                    dpdz = p.unsqueeze(0) * (torch.eye(p.size(0), device=dataset_probs.device) - p.unsqueeze(1))
                    dgdz = (dgdp_expanded * dpdz).sum(dim=0)
                    # 计算负熵正则项关于 dataset_probs 的导数
                    log_p = torch.log(p + 1e-8)  # 避免 log(0)
                    dHdp = -(1 + log_p)
                    dHdp_expanded = dHdp.unsqueeze(1).expand(-1, p.size(0))
                    dHdz = (dHdp_expanded * dpdz).sum(dim=0)
                    # 综合两部分导数
                    grad = dgdz + lambda_reg * dHdz
                    return grad
                
                lambda_reg = 0.00000000000 # 正则化系数
                grad = compute_dataset_probs_gradient(probs_module.probs, inner_products, lr, lambda_reg)
                dataset_optimizer.zero_grad()
                probs_module.probs.grad = grad
                fabric.print('The true gradient is:\n')
                fabric.print(grad)
                dataset_optimizer.step()
                fast_mixture_dir = os.path.join(out_dir, "FastMixtureOut")
                os.makedirs(fast_mixture_dir, exist_ok=True)
                # 保存probs_module
                probs_module_path = os.path.join(fast_mixture_dir, f"probs_module_step{state['iter_num']}.pt")
                torch.save(probs_module.state_dict(), probs_module_path)
                
                probs_softmax = torch.softmax(probs_module.probs, dim=-1).detach().cpu().squeeze().numpy()
                # 逐维度 log
                subset_names = [name for name, _ in train_data_config]
                log_dict = {f"probs_softmax/{subset_names[i]}": v for i, v in enumerate(probs_softmax)}
                wandb.log(log_dict, step=state["iter_num"])
                
            
            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                logits = model(input_ids).logits
                probs = torch.nn.functional.softmax(probs_module.probs, dim=0) # KKK
                sample_weights = probs[group_ids].to(fabric.device) # KKK
                loss = loss_func(logits, targets, sample_weights) # KKK
                # loss = chunked_cross_entropy(logits, targets, chunk_size=0)
                fabric.backward(loss / gradient_accumulation_steps)

            if not is_accumulating:
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                fabric.log_dict({
                    "gradient_norm": grad_norm.item()
                })
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
            elif fabric.device.type == "xla":
                xm.mark_step()
            state["iter_num"] += 1
            # input_id: B L 
            total_lengths += input_ids.size(1)
            t1 = time.perf_counter()
            fabric.print(
                    f"iter {state['iter_num']} learning rate {lr} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    # print days as well
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                )
    
            monitor.on_train_batch_end(
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                state["step_count"], 
                lengths=total_lengths,
                train_loss = loss.item()
            )
                
            if val_dataloaders is not None and not is_accumulating and state["step_count"] % eval_step_interval == 0:
                try:            
                    val_names = [name[0][0].replace("valid_", "") for name in val_data_config]
                except Exception as e:
                    print(e)
                    val_names = [str(i) for i in range(len(val_dataloaders))]
                for i in range(len(val_dataloaders)):
                    t0 = time.perf_counter()
                    val_loss = validate(fabric, model, val_dataloaders[i], val_names[i])
                    t1 = time.perf_counter() - t0
                    monitor.eval_end(t1)
                    fabric.print(f"step {state['iter_num']}: {val_names[i]} val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
                    fabric.log_dict({f"metric/{val_names[i]}_val_loss": val_loss.item(), "total_tokens": model.config.max_position_embeddings * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                    fabric.log_dict({f"metric/{val_names[i]}_val_ppl": math.exp(val_loss.item()), "total_tokens": model.config.max_position_embeddings * (state["iter_num"] + 1) * micro_batch_size * fabric.world_size}, state["step_count"])
                    fabric.barrier()

            if not is_accumulating and state["step_count"] % save_step_interval == 0:
                checkpoint_path = out_dir / f"iter-{state['step_count']:06d}-ckpt.pth"
                if fabric.global_rank == 0:
                    # delete all the checkpoints except the last one
                    delete_all_except_last(out_dir)
                    
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                if only_save_model:
                    saved_state = state["model"]
                else:
                    saved_state = state
                fabric.save(checkpoint_path, saved_state)
            

        
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, name: str = None) -> torch.Tensor:
    fabric.print(f"Validating {name} ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    k = -1
    for val_data, _ in val_dataloader:
        k = k + 1
        if k >= eval_iters:
            break
        input_ids = val_data[:, 0 : model.config.max_position_embeddings].contiguous()
        targets = val_data[:, 1 : model.config.max_position_embeddings + 1].contiguous()
        logits = model(input_ids).logits
        loss = chunked_cross_entropy(logits, targets, chunk_size=0)
    
        # loss_func = FusedCrossEntropyLoss()
        # loss = loss_func(logits, targets)
        losses[k] = loss.item()
    
    # print top 100 losses
    # fabric.print("top 100 losses: {}".format(losses[:100]))
    out = losses.mean()

    model.train()
    return out


def create_train_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    # check the validness
    for idx in range(len(data_config) - 1, -1, -1):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}_*")))     
        if len(filenames) < total_devices:
            fabric.print("skip dataset {}".format(prefix))
            del data_config[idx]
            continue

    for idx in range(len(data_config)):
        prefix = data_config[idx][0]
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}_*")))
        random.seed(seed)
        random.shuffle(filenames)
        fabric.print("create dataset {}".format(prefix))

        dataset = PackedDataset(
            filenames,
            # n_chunks control the buffer size. 
            # Note that the buffer size also impacts the random shuffle
            # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
            n_chunks=1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
        )

    weights = [weight for _, weight in data_config]
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_val_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    
    val_data_loaders = []
    
    # check the validness
    for idx in range(len(val_data_config) - 1, -1, -1):
        data_config = val_data_config[idx]
        delete_val_flag = False
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}_*")))
            if len(filenames) < total_devices:
                fabric.print("skip val dataset {}".format(prefix))
                delete_val_flag = True
                break
        if delete_val_flag:
            del val_data_config[idx]


    for data_config in val_data_config:
        datasets = []
        for prefix, _ in data_config:
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}_*")))
            random.seed(seed)
            random.shuffle(filenames)

            dataset = PackedDataset(
                filenames,
                # n_chunks control the buffer size. 
                # Note that the buffer size also impacts the random shuffle
                # (PackedDataset is an IterableDataset. So the shuffle is done by prefetch a buffer and shuffle the buffer)
                n_chunks=1,
                block_size=block_size,
                shuffle=shuffle,
                seed=seed+fabric.global_rank,
                num_processes=fabric.world_size,
                process_rank=fabric.global_rank,
            )
            datasets.append(dataset)

        if not datasets:
            raise RuntimeError(
                f"No data found at {data_dir}. Make sure you ran prepare_redpajama.py to create the dataset."
            )

        weights = [weight for _, weight in data_config]
        sum_weights = sum(weights)
        weights = [el / sum_weights for el in weights]

        check_flag = True
        for dataset in datasets:
            if len(dataset._filenames) == 0:
                check_flag = False
                break
        
        if check_flag:
            combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)
            val_data_loaders.append(DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True))
            fabric.print("create val dataset {}".format(data_config))
        else:
            fabric.print("there are something wrong with the val dataset {}".format(data_config))
    return val_data_loaders


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/redpajama_sample"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    # Increase by one because we need the next word as well
    effective_block_size = block_size + 1
    train_dataloader = create_train_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    # fabric.print("Check \n\n")
    # fabric.print(batch_size)
    # fabric.print("Check \n\n")
    val_dataloader = (
        create_val_dataloader(
            batch_size=batch_size//8,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
