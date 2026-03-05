"""The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task."""

import collections
import inspect
import math
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import pandas as pd

import transformers
from transformers.file_utils import is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.integrations import (
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_scheduler

from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_utils import (
    default_compute_objective,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import TrainOutput

from tqdm import tqdm, trange
from torch.optim import SGD
import torch.nn.functional as F

from src.linearhead_trainer import LinearHeadTrainer
from transformers.trainer_callback import TrainerState

import copy
import gc
from opacus.accountants.utils import get_noise_multiplier


_use_native_amp = False
_use_apex = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    from ray import tune

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

########## The above part is copied from Transformers' trainer (3.4.0) ##########

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]

    raise Exception("No metric founded for {}".format(metrics))

from functools import partial
import dataclasses as dc


def replace_parameters_with_view(module, predicate=lambda p: p.requires_grad):
    """
    Replaces model parameters with a single contiguous tensor view.
    Explicitly moves old tensor data to CPU to free up VRAM, using the original device for the new tensor.
    """
    param_info = []
    target_device = None

    def recurse(sub):
        nonlocal target_device
        for name, p in sub.named_parameters(recurse=False):
            if predicate(p):
                assert p.grad is None
                if target_device is None:
                    target_device = p.device
                param_info.append({
                    'module': sub,
                    'name': name,
                    'shape': p.shape,
                    'data_cpu': p.data.to('cpu'),
                    'numel': p.numel()
                })
                sub._parameters[name] = None
        for child in sub.children():
            recurse(child)
    recurse(module)

    gc.collect()
    torch.cuda.empty_cache()

    if not param_info:
        return None

    first_param_info = param_info[0]
    target_dtype = first_param_info['data_cpu'].dtype

    all_param_data_cpu = [info['data_cpu'].reshape(-1) for info in param_info]
    numel = sum(p.numel() for p in all_param_data_cpu)

    temp_flat_tensor_cpu = torch.empty(numel, device='cpu', dtype=target_dtype)
    offset = 0
    for data_cpu in all_param_data_cpu:
        temp_flat_tensor_cpu[offset:offset + data_cpu.numel()].copy_(data_cpu)
        offset += data_cpu.numel()

    p_full = nn.Parameter(temp_flat_tensor_cpu.to(target_device).requires_grad_(True))

    del all_param_data_cpu, temp_flat_tensor_cpu
    for info in param_info:
        del info['data_cpu']

    offset = 0
    with torch.no_grad():
        for info in param_info:
            p_shape = info['shape']
            p_numel = info['numel']
            p_view = p_full[offset:offset + p_numel].reshape(p_shape)
            info['module']._parameters[info['name']] = nn.Parameter(p_view)
            offset += p_numel

    param_info.clear()
    gc.collect()

    return p_full

def check_if_views(model, flat_tensor):
    """
    Checks if the model's parameters are views into the flat_tensor.
    """
    is_view = True
    flat_storage = flat_tensor.storage()
    offset = 0

    for name, param in model.named_parameters():
        param_storage = param.storage()
        
        # Check if the storage is the same
        if param_storage._data_ptr() != flat_storage._data_ptr():
            print(f"Parameter '{name}' is NOT a view. Its storage is different.")
            is_view = False
            break
            
        # Check if the offset is correct
        if (param.data_ptr() - flat_storage._data_ptr()) // param.itemsize != offset:
            print(f"Parameter '{name}' is a view, but its offset is incorrect.")
            is_view = False
            break
            
        offset += param.numel()

    if is_view:
        logger.info("✅ All model parameters are correctly set as views into the flattened tensor.")
    else:
        logger.info("❌ The model parameters are not all views as expected.")


class Trainer(LinearHeadTrainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.args.hf_inference_model:
            return

        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optimizer == 'adam':
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif self.args.optimizer == 'sgd':
                self.optimizer = SGD(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate
                )
            else:
                raise NotImplementedError
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    def should_optim(self, name, param):
        return (not self.args.layer_wise_optim or f".{self.state.global_step % self.model.config.num_hidden_layers}." in name) and param.requires_grad


    def zo_forward(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)
        if self.args.optimize_acc:
            loss, logits = model(**inputs)
            preds = F.softmax(logits, dim=-1)
            acc = torch.sum(torch.argmax(preds, 1) == inputs['labels']) / len(preds)
            loss = -acc
        else:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        self.state.zo_forward_step += 1
        return loss.detach()

    # Cone: set up first momentum vector as torch tensor
    def get_first_momentum(self, random_seed: int):
        torch.manual_seed(np.random.randint(1000000000))

        # compute total num elements
        total_numel = sum(p.data.numel() for _, p in self.named_parameters_to_optim)

        param_iter = iter(self.named_parameters_to_optim)
        name, param = next(param_iter)

        # one flat buffer
        self.momentum_flat = torch.empty(total_numel, device=param.data.device, dtype=param.data.dtype)

        # list of views
        self.momentum = []

        # First fill momentum with u_0 (0th random vector)

        offset = 0
        for name, param in self.named_parameters_to_optim:
            numel = param.data.numel()
            # generate random tensor of same shape
            rand = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            # copy into the corresponding slice of flat
            self.momentum_flat[offset:offset+numel].copy_(rand.view(-1))
            # create a view with the right shape
            view = self.momentum_flat[offset:offset+numel].view_as(param.data)
            self.momentum.append(view)
            offset += numel

        self.momentum_norm = np.sqrt(self.d)

        torch.manual_seed(random_seed)
        fac = np.sin(self.args.cone_theta)/(np.cos(self.args.cone_theta))
        for (name, param), mom in zip(self.named_parameters_to_optim, self.momentum):
            rand = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            mom.add_(rand, alpha=fac)
        


    def get_norm_momentum(self, projected_grad):
        fac = np.sqrt(self.d) / self.momentum_norm
        alpha = self.args.cone_beta/fac + (1-self.args.cone_beta)*projected_grad*np.cos(self.args.cone_theta)
        beta = (1-self.args.cone_beta)*projected_grad*np.sin(self.args.cone_theta)

        expected_squared_norm = self.d * (alpha**2 + beta**2)
        expected_norm = torch.sqrt(expected_squared_norm)

        return expected_norm
    
    def efficient_perturb_parameters(self, model: nn.Module, random_seed: int, scaling_factor=1):
        
        fac = np.sqrt(self.d) / self.momentum_norm
        alpha = fac * np.cos(self.args.cone_theta)
        with torch.no_grad():
            self.params_flat.add_(self.momentum_flat, alpha=scaling_factor*alpha*self.args.zero_order_eps)
        return model


    def get_num_samples(self):
        if self.args.zero_order_sample_scheduler is None:
            noise_sample_time = 1 
        elif self.args.zero_order_sample_scheduler == "linear":
            noise_sample_time = max(1, int(self.state.global_step / self.args.max_steps * self.args.zero_order_sample))
        elif self.args.zero_order_sample_scheduler == "constant":
            noise_sample_time = int(self.args.zero_order_sample)
        else:
            raise NotImplementedError

        return noise_sample_time

    def train(self, model_path=None, dev_objective=None, test_set=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        if self.args.from_linearhead and model_path is None:
            super().train(model_path, dev_objective) # Train output layer using LinearHeadTrainer

        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Flatten model parameters
        self.params_flat = replace_parameters_with_view(self.model)
        gc.collect()

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info(" Cone active = %s", str(self.args.cone))

        self.state = TrainerState()
        self.state.global_step = 0
        start_time = time.time()
        self.state.zo_forward_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.state.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.state.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.state.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.state.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.state.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        metrics = None

        # set up cone parameters
        self.momentum = None
        self.momentum_norm = 0
        self.old_seed = None
        self.old_beta = self.args.cone_beta
        self.args.cone_beta = 0.1
        self.d = sum(p.numel() for n, p in model.named_parameters() if self.should_optim(n, p))
        assert not self.args.zero_order_use_trainer_optim, "not compatible"
        assert self.get_num_samples()==1
        assert self.args.gradient_accumulation_steps==1
        assert self.args.efficient_zero_order
        assert self.args.cone, "use other trainer.py for mezo"
        assert self.args.zero_order_sample_scheduler is None
        assert self.args.zero_order_optim
        assert not self.args.zero_order_clip_grad, 'gradient clipping not implemented yet for non-trainer ZO'

        infos_to_plot_x = []
        infos_to_plot_training_loss = []
        infos_to_plot_eval_x = []
        infos_to_plot_eval_loss = []
        infos_to_plot_eval_acc = []
        infos_to_plot_test_x = []
        infos_to_plot_test_loss = []
        infos_to_plot_test_acc = []



        for epoch in range(epochs_trained, int(num_train_epochs)):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Iteration", disable=not self.is_local_process_zero())
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None


            for step, inputs in enumerate(epoch_iterator):


                if self.args.sync_embedding_layers:
                    assert model.module.model_type == 'opt', 'did not implement embedding layer synchronization for non-OPT models'
                    model.module.model.decoder.embed_tokens.weight = model.module.lm_head.weight
                
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                    # Get parameters that should be optimized (for layer-wise optimization and prefix-tuning)
                self.named_parameters_to_optim = []
                for name, param in model.named_parameters():
                    if self.should_optim(name, param):
                        self.named_parameters_to_optim.append((name, param))
                    
                random_seed = np.random.randint(1000000000)
                
                # Cone: Set up first momentum vector
                if self.momentum is None:
                    self.get_first_momentum(random_seed)
                    self.old_seed = random_seed
                else:
                    self.update_momentum(self.old_seed, random_seed, projected_grad)
                    self.old_seed = random_seed
                    
                    if self.state.global_step <= 100:
                        self.args.cone_beta = 0.1
                    elif self.state.global_step <= 1000:
                        p = (self.state.global_step - 100) / 900.0
                        p = p ** 1.8  # ease-in softness
        
                        beta_start = 0.1
                        beta_target = self.old_beta
                        a, k = 8.0, 3.0
        
                        self.args.cone_beta = beta_target - (beta_target - beta_start) / ((1.0 + a * p) ** k)
                    else:
                        self.args.cone_beta = self.old_beta


                with torch.no_grad():
                    # first function evaluation
                    model = self.efficient_perturb_parameters(model, random_seed)
                    loss1 = self.zo_forward(model, inputs)

                    # second function evaluation
                    model = self.efficient_perturb_parameters(model, random_seed, scaling_factor=-2)
                    loss2 = self.zo_forward(model, inputs)

                projected_grad = (loss1 - loss2) / (2 * self.args.zero_order_eps)

                # reset model back to its parameters at start of step
                model = self.efficient_perturb_parameters(model, random_seed)

                # apply gradient updates
                

                torch.manual_seed(random_seed)

                fac = np.sqrt(self.d) / self.momentum_norm
                alpha = fac * np.cos(self.args.cone_theta)
                with torch.no_grad():
                    self.params_flat.sub_(self.momentum_flat, alpha=self.args.learning_rate*projected_grad*alpha)

                if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                        self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        logs["loss"] = loss1.item()
                        logs["learning_rate"] = self.args.learning_rate
                        logs["beta"] = self.args.cone_beta
                        logs["theta"] = self.args.cone_theta
                        logs["last_grad"] = projected_grad.item()
                        logs["global_step"] = self.state.global_step
                        logs["zo_forward_step"] = self.state.zo_forward_step
                        logs["max_steps"] = self.args.max_steps
                        logs["max_zo_forward_steps"] = self.args.max_zo_forward_steps
                        logs["time"] = int(time.time() - start_time)
                        self.log(logs)
                        logger.info(str(logs))
                        # Plot loss for epochs
                        infos_to_plot_x.append(logs["global_step"])
                        infos_to_plot_training_loss.append(logs["loss"])


                self.state.global_step += 1
                self.epoch = epoch + (step + 1) / len(epoch_iterator)
                

                        


                if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                    epoch_iterator.close()
                    break
                if self.state.global_step % self.args.eval_steps == 0:
                    output = self.evaluate()
                    metrics = output.metrics

                    # Cone: Save inform which should be plotted
                    infos_to_plot_eval_x.append(self.state.global_step)
                    if test_set.args.task_name=='mnli':
                        infos_to_plot_eval_acc.append(metrics["eval_mnli/acc"])
                        infos_to_plot_eval_loss.append(metrics["eval_loss"])
                    else:
                        infos_to_plot_eval_acc.append(metrics["eval_acc"])
                        infos_to_plot_eval_loss.append(metrics["eval_loss"])

                    objective = self.dev_objective(metrics)
                    if objective > self.objective:
                        logger.info("Best dev result: {}".format(objective))
                        self.objective = objective

                        # Now we save this to (CPU) memory instead of disk <-- much faster
                        self.best_model_ckpt = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                        # uses this model for final test_acc

                    # Cone: Now log test accuracy and loss
                    logger.info('** test **')
                    test_output = self.evaluate(eval_dataset=test_set)
                    test_metrics = test_output.metrics

                    infos_to_plot_test_x.append(self.state.global_step)

                    if test_set.args.task_name=='mnli':
                        infos_to_plot_test_acc.append(test_metrics["eval_mnli/acc"])
                        infos_to_plot_test_loss.append(test_metrics["eval_loss"])
                    else:
                        infos_to_plot_test_acc.append(test_metrics["eval_acc"])
                        infos_to_plot_test_loss.append(test_metrics["eval_loss"])


            if self.args.max_steps > 0 and self.state.global_step > self.args.max_steps or (self.args.max_zo_forward_steps > 0 and self.state.zo_forward_step > self.args.max_zo_forward_steps):
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        

        train_outp = TrainOutput(self.state.global_step, tr_loss / self.state.global_step, metrics)

        infos_to_plot_eval_x.append(self.state.global_step)

        if test_set.args.task_name=='mnli':
            infos_to_plot_eval_acc.append(train_outp.metrics["eval_mnli/acc"])
            infos_to_plot_eval_loss.append(train_outp.metrics["eval_loss"])
        else:
            infos_to_plot_eval_acc.append(train_outp.metrics["eval_acc"])
            infos_to_plot_eval_loss.append(train_outp.metrics["eval_loss"])

        if self.args.cone:
            pic_title = f'theta={self.args.cone_theta}, beta={self.args.cone_beta}, lr={self.args.learning_rate}, its={self.args.max_steps}'
        else:
            pic_title = f'MeZO, lr={self.args.learning_rate},  its={self.args.max_steps}'

        # Save log to csv
        df = pd.DataFrame({'steps': infos_to_plot_x, 'train_loss': infos_to_plot_training_loss})
        df['eval_acc'] = pd.NA
        df['eval_loss'] = pd.NA

        for x, y, z in zip(infos_to_plot_eval_x, infos_to_plot_eval_loss, infos_to_plot_eval_acc):
            df.loc[df['steps'] == x, 'eval_loss'] = y
            df.loc[df['steps'] == x, 'eval_acc'] = z

        df['test_acc'] = pd.NA
        df['test_loss'] = pd.NA

        for x, y, z in zip(infos_to_plot_test_x, infos_to_plot_test_loss, infos_to_plot_test_acc):
            df.loc[df['steps'] == x, 'test_loss'] = y
            df.loc[df['steps'] == x, 'test_acc'] = z

        return train_outp, self.objective, df, None, pic_title


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description="Evaluation")

        self.log(output.metrics)
        logger.info(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
    
    def update_momentum(self, random_seed, new_random_seed, projected_grad):
        # previously at end of iteration:
        torch.manual_seed(random_seed)
        fac = np.sqrt(self.d) / self.momentum_norm
        a = (1-self.args.cone_beta) * projected_grad * fac * np.cos(self.args.cone_theta)
        k = np.sin(self.args.cone_theta)/(np.cos(self.args.cone_theta) * fac)
        self.momentum_flat.mul_(a + self.args.cone_beta)
        
        for (name, param), momentum_view in zip(self.named_parameters_to_optim, self.momentum):
            u = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            momentum_view.add_(u, alpha=-self.args.cone_beta*k)

        # mom = b * u0 + (1-b) * pg * [cos(t)*fac*u0 + sin(t)*u_1]
        # Update momentum norm and reseed for next iteration
        self.momentum_norm = self.get_norm_momentum(projected_grad)
        torch.manual_seed(new_random_seed)
        fac = np.sqrt(self.d) / self.momentum_norm
        fac2 = np.sin(self.args.cone_theta)/(np.cos(self.args.cone_theta) * fac)
        for (name, param), mom in zip(self.named_parameters_to_optim, self.momentum):
            rand = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            mom.add_(rand, alpha=fac2)
        
