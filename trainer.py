"""
Main training loop and training logic.
"""
import os
import time
import sys
import math
import json
from typing import Dict
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

try:
    import torch._inductor.config as inductor_config
except Exception:
    inductor_config = None

from models.nanoGPT import GPT, GPTConfig
from args import TrainingConfig
from data_loader import DistributedDataLoader
from optimizers import create_optimizers, create_lr_schedulers
from utils import Logger, calculate_steps, get_memory_usage
from train_metrics import ActivationMonitor, ParamUpdateMonitor


class Trainer:
    """
    Main trainer class that handles the training loop and validation.
    """
    
    def __init__(self, config: TrainingConfig, ddp_rank: int, ddp_local_rank: int, 
                 ddp_world_size: int, device: str, master_process: bool):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
            ddp_rank: Distributed data parallel rank
            ddp_local_rank: Local rank within node
            ddp_world_size: Total number of processes
            device: Device string (e.g., 'cuda:0')
            master_process: Whether this is the master process
        """
        self.config = config
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        
        # Initialize logger
        self.logger = Logger(config, master_process)
        
        # Calculate training parameters
        self.val_steps, self.train_accumulation_steps = calculate_steps(config, ddp_world_size)
        
        # Initialize data loaders
        self._setup_data_loaders()
        
        # Initialize model
        self._setup_model()
        
        # Initialize optimizers and schedulers
        self._setup_optimizers()

        # Optional: stability monitors (only on master to reduce overhead)
        self.activation_monitor = None
        self.param_update_monitor = None
        if self.master_process:
            if getattr(self.config, "log_activation_norm", False) or getattr(self.config, "log_activation_update_norm", False):
                try:
                    blocks = self.raw_model.transformer.h
                    self.activation_monitor = ActivationMonitor(
                        blocks=blocks,
                        embedding_module=self.raw_model.transformer.wte,
                        lm_head_module=self.raw_model.lm_head,
                        device=torch.device(self.device),
                        log_activation_norm=getattr(self.config, "log_activation_norm", False),
                        log_activation_update_norm=getattr(self.config, "log_activation_update_norm", False),
                    )
                except Exception as e:
                    print(f"Warning: could not enable activation monitoring: {e}")
                    self.activation_monitor = None

            if getattr(self.config, "log_param_update_norm", False):
                try:
                    self.param_update_monitor = ParamUpdateMonitor(
                        optimizers=self.optimizers,
                        device=torch.device(self.device),
                    )
                except Exception as e:
                    print(f"Warning: could not enable param update monitoring: {e}")
                    self.param_update_monitor = None
        
        # Get code for logging
        self.code = self._get_training_code()
        
        # Initialize wandb
        self.logger.init_wandb()

        # Initialize swanlab
        self.logger.init_swanlab()

        # Note: if we run evaluation during training, prefer distributed eval across ranks
        # to avoid having rank0 pause everyone else for a long time.
    
    def _setup_data_loaders(self):
        """Set up training and validation data loaders."""
        B, T = self.config.device_batch_size, self.config.sequence_length
        
        self.train_loader = DistributedDataLoader(
            self.config.input_bin, B, T, self.ddp_rank, self.ddp_world_size
        )
        self.val_loader = DistributedDataLoader(
            self.config.input_val_bin, B, T, self.ddp_rank, self.ddp_world_size
        )
        
        if self.master_process:
            print(f"Training DataLoader: total number of tokens: {self.train_loader.ntok_total} "
                  f"across {len(self.train_loader.files)} files")
            print(f"Validation DataLoader: total number of tokens: {self.val_loader.ntok_total} "
                  f"across {len(self.val_loader.files)} files")
    
    def _setup_model(self):
        """Set up the model with compilation and DDP wrapping."""
        model_config = GPTConfig(
            vocab_size=self.config.vocab_size,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd
        )
        
        model = GPT(model_config)
        model = model.cuda()
        
        # Enable coordinate-descent tuning when this torch build exposes it.
        if inductor_config is not None and hasattr(inductor_config, "coordinate_descent_tuning"):
            inductor_config.coordinate_descent_tuning = True
        
        model = torch.compile(model)
        
        # Wrap model in DDP
        self.model = DDP(model, device_ids=[self.ddp_local_rank])
        self.raw_model = self.model.module  # unwrapped model
        
        # Set up autocast context
        self.ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
    
    def _setup_optimizers(self):
        """Set up optimizers and learning rate schedulers."""
        self.optimizers = create_optimizers(
            self.config, self.raw_model, self.ddp_rank, self.ddp_world_size
        )
        self.schedulers = create_lr_schedulers(self.optimizers, self.config)
    
    def _get_training_code(self):
        """Get the training script code for logging."""
        try:
            with open(sys.argv[0]) as f:
                return f.read()
        except Exception:
            return "Could not read training script code"
    
    def _get_lr(self, step: int) -> float:
        """Get current learning rate based on step."""
        if step >= self.config.num_iterations:
            return 0.0
        
        if getattr(self.config, 'lr_scheduler', 'trapezoidal') == 'cosine':
             # 1) linear warmup for warmup_iters steps
            if step < self.config.warmup_iters:
                return (step + 1) / self.config.warmup_iters
            # 2) cosine decay
            decay_ratio = (step - self.config.warmup_iters) / (self.config.num_iterations - self.config.warmup_iters)
            return 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        else:
            # Trapezoidal (default)
            # 1) linear warmup for warmup_iters steps
            if step < self.config.warmup_iters:
                return (step + 1) / self.config.warmup_iters
            # 2) constant lr for a while
            elif step < self.config.num_iterations - self.config.warmdown_iters:
                return 1.0
            # 3) linear warmdown
            else:
                decay_ratio = (self.config.num_iterations - step) / self.config.warmdown_iters
                return decay_ratio

    def validate(self) -> float:
        """
        Run validation and return validation loss.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        self.val_loader.reset()
        val_loss = 0.0

        with torch.no_grad():
            for _ in range(self.val_steps):
                x_val, y_val = self.val_loader.next_batch()
                with self.ctx:
                    _, loss = self.model(x_val, y_val, return_logits=False)
                    val_loss += loss.detach()
                    del loss
        
        # All-reduce validation loss across processes
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss /= self.val_steps
        
        return val_loss
    
    def train_step(self, step: int):
        """
        Perform one training step with gradient accumulation.
        
        Returns:
            Training loss tensor
        """
        self.model.train()

        extra_metrics = {}
        capture_act = (
            self.activation_monitor is not None
            and getattr(self.config, "activation_log_every", 1) > 0
            and (step % int(getattr(self.config, "activation_log_every", 1)) == 0)
        )
        if self.activation_monitor is not None:
            self.activation_monitor.begin_step(capture_enabled=capture_act)

        q_stat_accumulators = []
        for opt in self.optimizers:
            fn = getattr(opt, "accumulate_q_statistics_from_current_graph", None)
            if callable(fn):
                q_stat_accumulators.append(fn)
        needs_q_stats = len(q_stat_accumulators) > 0
        
        for i in range(1, self.train_accumulation_steps + 1):
            # Get batch data
            x, y = self.train_loader.next_batch()
            
            # Forward pass
            with self.ctx:
                _, loss = self.model(x, y, return_logits=False)
                train_loss = loss.detach()
            
            # Backward pass
            if i < self.train_accumulation_steps:
                with self.model.no_sync():  # No gradient sync until last step
                    loss.backward()
            else:
                loss.backward()  # Sync gradients on last step

            if needs_q_stats:
                # Keep this hook path for any future optimizer that needs live-graph stats.
                with self.ctx:
                    q_model = self.raw_model
                    if hasattr(q_model, "_orig_mod"):
                        q_model = q_model._orig_mod
                    _, loss_for_q = q_model(x, y, return_logits=False)
                del loss_for_q
                for accumulate_q in q_stat_accumulators:
                    accumulate_q()
        
        # Scale gradients by accumulation steps
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad /= self.train_accumulation_steps

        capture_update = (
            self.param_update_monitor is not None
            and getattr(self.config, "param_update_norm_every", 1) > 0
            and (step % int(getattr(self.config, "param_update_norm_every", 1)) == 0)
        )
        if self.param_update_monitor is not None:
            self.param_update_monitor.set_capture_enabled(capture_update)
        
        # Step optimizers and schedulers
        for opt, sched in zip(self.optimizers, self.schedulers):
            opt.step()
            sched.step()

        if capture_update:
            extra_metrics.update(self.param_update_monitor.capture_post_and_compute())
        
        # Clear gradients
        self.model.zero_grad(set_to_none=True)

        if self.activation_monitor is not None:
            extra_metrics.update(self.activation_monitor.end_step())
        
        for opt in self.optimizers:
            if opt.track_update_stats:
                extra_metrics.update(opt.last_update_details)

        return train_loss, extra_metrics
    
    def train(self):
        """Main training loop."""
        training_time_ms = 0
        
        # Start timing
        with record_function("sync"):
            torch.cuda.synchronize()
        t0 = time.time()
        
        # Reset data loader
        self.train_loader.reset()
        
        use_profiler = getattr(self.config, 'profile', False) and (not getattr(self.config, 'profile_rank0_only', True) or self.master_process)
        if use_profiler:
            trace_dir = os.path.join(self.config.profile_dir, f"rank{self.ddp_rank}")
            os.makedirs(trace_dir, exist_ok=True)
            prof_schedule = schedule(wait=self.config.profile_wait,
                                     warmup=self.config.profile_warmup,
                                     active=self.config.profile_active,
                                     repeat=self.config.profile_repeat)
            prof_activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            profiler_ctx = profile(
                activities=prof_activities,
                schedule=prof_schedule,
                on_trace_ready=tensorboard_trace_handler(trace_dir),
                record_shapes=True,
                profile_memory=self.config.profile_memory,
                with_stack=self.config.profile_with_stack,
            )
        else:
            profiler_ctx = None

        def _training_loop_body():
            nonlocal training_time_ms, t0
            for step in tqdm(range(self.config.num_iterations + 1), desc="Training Steps"):
                last_step = (step == self.config.num_iterations)
                
                # Reset timing after first 10 steps (which are slower for various reasons)
                if step == 10:
                    training_time_ms = 0
                    t0 = time.time()
                
                # Calculate timed steps (excluding first 10 steps)
                timed_steps = float('nan') if step <= 11 else (step - 10) + 1
                
                # Validation
                if last_step or (self.config.val_loss_every > 0 and step % self.config.val_loss_every == 0):
                    # Stop training timer
                    with record_function("sync"):
                        torch.cuda.synchronize()
                    training_time_ms += 1000 * (time.time() - t0)
                    
                    # Run validation
                    with record_function("validation"):
                        val_loss = self.validate()
                    
                    # Log validation results
                    if self.master_process:
                        # Log to wandb with additional metrics
                        if self.logger.wandb_initialized:
                            wandb_metrics = {
                                "val_loss": val_loss.item(),
                            }
                            self.logger.log_step(step, wandb_metrics)
                        
                        # Log to swanlab with additional metrics
                        if self.logger.swanlab_initialized:
                            swanlab_metrics = {
                                "val_loss": val_loss.item(),
                            }
                            self.logger.log_step(step, swanlab_metrics)
                    
                    # Optional: run selected eval tasks after each validation.
                    eval_during_train_metrics = self._run_eval_during_train(step=int(step))
                    if self.master_process and eval_during_train_metrics:
                        self.logger.log_step(int(step), eval_during_train_metrics, prefix="eval")

                    # Restart training timer
                    with record_function("sync"):
                        torch.cuda.synchronize()
                    t0 = time.time()
                
                # Save checkpoint
                if self.master_process and (last_step or (self.config.save_every > 0 and step % self.config.save_every == 0)):
                    # Stop training timer
                    with record_function("sync"):
                        torch.cuda.synchronize()
                    training_time_ms += 1000 * (time.time() - t0)
                    
                    # Save checkpoint
                    optimizer_states = [opt.state_dict() for opt in self.optimizers]
                    self.logger.save_checkpoint(
                        step,
                        self.code,
                        self.raw_model.state_dict(),
                        optimizer_states,
                        training_config=vars(self.config),
                        model_config=vars(self.raw_model.config) if hasattr(self.raw_model, "config") else None,
                    )
                    
                    # Restart training timer
                    with record_function("sync"):
                        torch.cuda.synchronize()
                    t0 = time.time()
                
                # Break after validation/checkpoint on last step
                if last_step:
                    break
                
                # Training step
                with record_function("train_iteration"):
                    train_loss, extra_metrics = self.train_step(step)
                
                # Log training progress
                if self.master_process:
                    approx_time = training_time_ms + 1000 * (time.time() - t0)
                    step_avg = approx_time / timed_steps if timed_steps > 0 else 0
                    current_lr = self._get_lr(step)
                    
                    # Log to wandb
                    if self.logger.wandb_initialized:
                        wandb_metrics = {
                            "train_loss": train_loss.item(),
                            "learning_rate": current_lr,
                            "training_time_ms": training_time_ms,
                            "step_avg_time_ms": step_avg,
                        }
                        if extra_metrics:
                            wandb_metrics.update(extra_metrics)
                        self.logger.log_step(step + 1, wandb_metrics)
                    
                    # Log to swanlab with additional metrics
                    with record_function("swanlab_log"):
                        if self.logger.swanlab_initialized:
                            swanlab_metrics = {
                                "train_loss": train_loss.item(),
                                "learning_rate": current_lr,
                                "training_time_ms": training_time_ms,
                                "step_avg_time_ms": step_avg,
                            }
                            if extra_metrics:
                                swanlab_metrics.update(extra_metrics)
                            self.logger.log_step(step + 1, swanlab_metrics)
                
                # Advance profiler step
                if use_profiler:
                    prof.step()

        if use_profiler:
            with profiler_ctx as prof:
                _training_loop_body()
        else:
            _training_loop_body()

        # Eval-after-train (rank0 only).
        #
        # Multi-node note:
        # We must keep *all* ranks alive until rank0 completes evaluation; otherwise,
        # job-launchers that run multiple `torchrun` invocations sequentially (e.g. `run.sh`)
        # can desynchronize across nodes and the next rendezvous can time out with:
        #   DistStoreError: Timed out ... 1/N clients joined.
        #
        # Do NOT use a file-based barrier unless `logs/` is a shared filesystem across nodes.
        # Here we use a long-timeout barrier group so non-rank0 workers can safely wait for
        # long-running evals without desynchronizing multi-node jobs.
        if getattr(self.config, "eval_after_train", False):
            sync_pg = None
            if dist.is_available() and dist.is_initialized() and self.ddp_world_size > 1:
                # 12 hours default; override with --eval_after_train_timeout_seconds if needed.
                timeout_s = int(getattr(self.config, "eval_after_train_timeout_seconds", 12 * 60 * 60))
                # Prefer NCCL here: some clusters block Gloo's TCP transport, while NCCL is already
                # configured for multi-node GPU training.
                sync_pg = dist.new_group(backend="nccl", timeout=timedelta(seconds=timeout_s))

            if self.master_process:
                try:
                    self._run_eval_after_train(step=self.config.num_iterations)
                except Exception as e:
                    print(f"Warning: eval_after_train failed: {e}")

            # Cross-node synchronization: ensure all ranks wait for rank0 eval to finish.
            if sync_pg is not None:
                dist.barrier(group=sync_pg)
                # Best-effort cleanup; the default process group is destroyed in main.py.
                try:
                    dist.destroy_process_group(sync_pg)
                except Exception:
                    pass

        # Final logging
        if self.master_process:
            peak_memory = get_memory_usage()
            print(f"peak memory consumption: {peak_memory} MiB")

    def finish(self) -> None:
        """Finish logging (wandb/swanlab) on master rank."""
        if self.master_process:
            self.logger.finish()

    def _eval_autocast_dtype(self):
        """Map config.eval_dtype -> torch.dtype | None (None means fp32/no autocast)."""
        dtype = getattr(self.config, "eval_dtype", "bf16")
        if dtype == "fp32":
            return None
        if dtype in ("bf16", "auto"):
            return torch.bfloat16
        if dtype == "fp16":
            return torch.float16
        return torch.bfloat16

    def _get_eval_model(self):
        """
        Return a model instance that supports `forward_logits()` for repo-local evals.
        Handles common wrappers like torch.compile (`_orig_mod`).
        """
        model = self.raw_model
        if not hasattr(model, "forward_logits") and hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _run_mmlu_fineweb(self, *, distributed: bool) -> dict:
        """
        Run FineWeb blog-style MMLU eval (0-shot, full-answer targets).

        If `distributed=True`, shard subjects across ranks and all-reduce totals.
        """
        from evals.tasks.mmlu_fineweb import run as run_mmlu_fineweb

        model = self._get_eval_model()
        model.eval()

        device = torch.device(self.device)
        autocast_dtype = self._eval_autocast_dtype()
        max_seq_len = int(
            getattr(self.config, "eval_max_seq_len", None) or getattr(self.config, "sequence_length", 1024)
        )
        subjects = str(getattr(self.config, "eval_mmlu_subjects", "all"))
        limit = getattr(self.config, "eval_limit", None)

        return run_mmlu_fineweb(
            model=model,
            device=device,
            autocast_dtype=autocast_dtype,
            max_seq_len=max_seq_len,
            subjects=subjects,
            limit=limit,
            ddp_rank=int(self.ddp_rank) if distributed else None,
            ddp_world_size=int(self.ddp_world_size) if distributed else None,
        )
    
    def _get_eval_during_train_tasks(self) -> list[str]:
        tasks = [t.strip() for t in str(getattr(self.config, "eval_during_train_tasks", "") or "").split(",") if t.strip()]
        # Deduplicate while preserving order.
        deduped: list[str] = []
        seen = set()
        for t in tasks:
            if t not in seen:
                deduped.append(t)
                seen.add(t)
        return deduped

    def _run_eval_during_train(self, *, step: int) -> Dict[str, float]:
        """
        Run selected eval tasks right after each validation step.

        This is called on all ranks. For multi-node runs, task runners use distributed
        reduction so no extra synchronization group is required.
        """
        tasks = self._get_eval_during_train_tasks()
        if not tasks:
            return {}

        model = self._get_eval_model()
        model.eval()

        device = torch.device(self.device)
        autocast_dtype = self._eval_autocast_dtype()
        max_seq_len = int(getattr(self.config, "eval_max_seq_len", None) or getattr(self.config, "sequence_length", 1024))
        distributed_eval = dist.is_available() and dist.is_initialized() and int(self.ddp_world_size) > 1
        ddp_rank = int(self.ddp_rank) if distributed_eval else None
        ddp_world_size = int(self.ddp_world_size) if distributed_eval else None

        eval_limit = getattr(self.config, "eval_limit", None)
        out: Dict[str, float] = {}

        for t in tasks:
            if t == "mmlu_fineweb":
                res = self._run_mmlu_fineweb(distributed=bool(distributed_eval))
                acc = float(res["overall"]["acc"])
                acc_norm = float(res["overall"]["acc_norm"])
                out["eval/mmlu_fineweb_acc"] = acc
                out["eval/mmlu_fineweb_acc_norm"] = acc_norm
                out["e/mmlu/acc"] = acc
                out["e/mmlu/acc_norm"] = acc_norm

            elif t == "mmlu":
                from evals.tasks.mmlu import run as run_mmlu

                res = run_mmlu(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    nshot=int(getattr(self.config, "eval_mmlu_nshot", 5)),
                    subjects=str(getattr(self.config, "eval_mmlu_subjects", "all")),
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/mmlu_accuracy"] = float(res["overall"]["accuracy"])
                out["eval/mmlu_accuracy_norm"] = float(res["overall"]["accuracy_norm"])

            elif t == "hellaswag":
                from evals.tasks.hellaswag import run as run_hellaswag

                res = run_hellaswag(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    split="validation",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/hellaswag_accuracy"] = float(res["accuracy"])
                out["e/hellaswag/acc_norm"] = float(res["acc_norm"])

            elif t == "arc_easy":
                from evals.tasks.arc import run as run_arc

                res = run_arc(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    challenge=False,
                    split="test",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/arc_easy_accuracy"] = float(res["accuracy"])
                out["e/arc_easy/acc_norm"] = float(res["acc_norm"])

            elif t in ("arc_challenge", "arc"):
                from evals.tasks.arc import run as run_arc

                res = run_arc(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    challenge=True,
                    split="test",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/arc_challenge_accuracy"] = float(res["accuracy"])
                out["e/arc/acc_norm"] = float(res["acc_norm"])

            elif t == "piqa":
                from evals.tasks.piqa import run as run_piqa

                res = run_piqa(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    split="validation",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/piqa_accuracy"] = float(res["accuracy"])
                out["eval/piqa_accuracy_norm"] = float(res["accuracy_norm"])
                out["e/piqa/acc_norm"] = float(res["acc_norm"])

            elif t == "openbookqa":
                from evals.tasks.openbookqa import run as run_openbookqa

                res = run_openbookqa(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    split="test",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/openbookqa_acc_norm"] = float(res["acc_norm"])
                out["e/openbookqa/acc_norm"] = float(res["acc_norm"])

            elif t == "commonsense_qa":
                from evals.tasks.commonsense_qa import run as run_commonsense_qa

                res = run_commonsense_qa(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    split="validation",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/commonsense_qa_acc_norm"] = float(res["acc_norm"])
                out["e/commonsense_qa/acc_norm"] = float(res["acc_norm"])

            elif t == "siqa":
                from evals.tasks.siqa import run as run_siqa

                res = run_siqa(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    split="validation",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/siqa_acc_norm"] = float(res["acc_norm"])
                out["e/siqa/acc_norm"] = float(res["acc_norm"])

            elif t == "winogrande":
                from evals.tasks.winogrande import run as run_winogrande

                res = run_winogrande(
                    model=model,
                    device=device,
                    autocast_dtype=autocast_dtype,
                    max_seq_len=max_seq_len,
                    config="winogrande_xl",
                    split="validation",
                    limit=eval_limit,
                    ddp_rank=ddp_rank,
                    ddp_world_size=ddp_world_size,
                )
                out["eval/winogrande_accuracy"] = float(res["accuracy"])
                out["e/winogrande/acc_norm"] = float(res["acc_norm"])

            elif t == "fineweb":
                from evals.tasks.arc import run as run_arc
                from evals.tasks.commonsense_qa import run as run_commonsense_qa
                from evals.tasks.hellaswag import run as run_hellaswag
                from evals.tasks.mmlu_fineweb import run as run_mmlu_fineweb
                from evals.tasks.openbookqa import run as run_openbookqa
                from evals.tasks.piqa import run as run_piqa
                from evals.tasks.siqa import run as run_siqa
                from evals.tasks.winogrande import run as run_winogrande

                fw_limit = eval_limit if eval_limit is not None else 1000
                metrics: Dict[str, float] = {}
                metrics["commonsense_qa/acc_norm"] = float(
                    run_commonsense_qa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["hellaswag/acc_norm"] = float(
                    run_hellaswag(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["openbookqa/acc_norm"] = float(
                    run_openbookqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="test",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["piqa/acc_norm"] = float(
                    run_piqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["siqa/acc_norm"] = float(
                    run_siqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["winogrande/acc_norm"] = float(
                    run_winogrande(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        config="winogrande_xl",
                        split="validation",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["arc/acc_norm"] = float(
                    run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=True,
                        split="test",
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["acc_norm"]
                )
                metrics["mmlu/acc_norm"] = float(
                    run_mmlu_fineweb(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        subjects=str(getattr(self.config, "eval_mmlu_subjects", "all")),
                        limit=fw_limit,
                        ddp_rank=ddp_rank,
                        ddp_world_size=ddp_world_size,
                    )["overall"]["acc_norm"]
                )
                agg = sum(metrics.values()) / len(metrics)
                out["e/agg_score"] = float(agg)
                for k, v in metrics.items():
                    out[f"e/{k}"] = float(v)

            else:
                raise ValueError(f"Unknown task in --eval_during_train_tasks: {t}")

        return out

    def _run_eval_after_train(self, *, step: int) -> None:
        """
        Run eval tasks at the end of training (rank0 only).

        This is intentionally run on rank0 only.
        """
        if not self.master_process:
            return

        tasks_str = str(getattr(self.config, "eval_tasks", "") or "")
        tasks = [t.strip() for t in tasks_str.split(",") if t.strip()]
        if not tasks:
            return

        # Expand "bench" into a concrete task list.
        if "bench" in tasks:
            inner = [
                "mmlu",
                "hellaswag",
                "arc_easy",
                "arc_challenge",
                "piqa",
                "winogrande",
                "openbookqa",
                "siqa",
                "commonsense_qa",
            ]
            expanded = []
            for t in tasks:
                if t == "bench":
                    expanded.extend(inner)
                else:
                    expanded.append(t)
            # Deduplicate while preserving order.
            tasks = list(dict.fromkeys(expanded))

        model = self._get_eval_model()
        model.eval()

        device = torch.device(self.device)
        autocast_dtype = self._eval_autocast_dtype()
        max_seq_len = int(getattr(self.config, "eval_max_seq_len", None) or getattr(self.config, "sequence_length", 1024))
        eval_limit = getattr(self.config, "eval_limit", None)

        out_dir = None
        if getattr(self.logger, "logdir", None):
            out_dir = os.path.join(self.logger.logdir, "eval")
            os.makedirs(out_dir, exist_ok=True)

        print(f"[eval_after_train] tasks={tasks} device={device} dtype={autocast_dtype} max_seq_len={max_seq_len}")

        all_results = []
        log_metrics = {}

        for t in tasks:
            try:
                if t == "pretrain":
                    from evals.pretrain import run as run_pretrain

                    res = run_pretrain(
                        model=model,
                        input_bin=str(getattr(self.config, "input_val_bin")),
                        batch_size=int(getattr(self.config, "device_batch_size")),
                        sequence_length=int(getattr(self.config, "sequence_length")),
                        device=device,
                        autocast_dtype=autocast_dtype,
                        eval_tokens=int(getattr(self.config, "eval_pretrain_tokens", 1024 * 1024)),
                    )
                    all_results.append(res)
                    log_metrics.update(
                        {
                            "eval/pretrain_loss": float(res["loss"]),
                            "eval/pretrain_ppl": float(res["perplexity"]),
                            "eval/pretrain_token_acc": float(res["token_accuracy"]),
                            "eval/pretrain_tokens_per_s": float(res["tokens_per_second"]),
                        }
                    )

                elif t == "mmlu":
                    from evals.tasks.mmlu import run as run_mmlu

                    res = run_mmlu(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        nshot=int(getattr(self.config, "eval_mmlu_nshot", 5)),
                        subjects=str(getattr(self.config, "eval_mmlu_subjects", "all")),
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/mmlu_accuracy"] = float(res["overall"]["accuracy"])
                    log_metrics["eval/mmlu_num_subjects"] = int(len(res.get("evaluated_subjects", [])))
                    log_metrics["eval/mmlu_num_skipped_subjects"] = int(len(res.get("skipped_subjects", [])))

                elif t == "mmlu_fineweb":
                    # Eval-after-train is rank0-only by design.
                    res = self._run_mmlu_fineweb(distributed=False)
                    all_results.append(res)
                    acc = float(res["overall"]["acc"])
                    acc_norm = float(res["overall"]["acc_norm"])
                    log_metrics["eval/mmlu_fineweb_acc"] = acc
                    log_metrics["eval/mmlu_fineweb_acc_norm"] = acc_norm
                    # FineWeb-style tags (matches their eval_results.csv column names).
                    log_metrics["e/mmlu/acc"] = acc
                    log_metrics["e/mmlu/acc_norm"] = acc_norm

                elif t == "hellaswag":
                    from evals.tasks.hellaswag import run as run_hellaswag

                    res = run_hellaswag(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/hellaswag_accuracy"] = float(res["accuracy"])

                elif t == "arc_easy":
                    from evals.tasks.arc import run as run_arc

                    res = run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=False,
                        split="test",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/arc_easy_accuracy"] = float(res["accuracy"])

                elif t == "arc_challenge":
                    from evals.tasks.arc import run as run_arc

                    res = run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=True,
                        split="test",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/arc_challenge_accuracy"] = float(res["accuracy"])

                elif t == "piqa":
                    from evals.tasks.piqa import run as run_piqa

                    res = run_piqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/piqa_accuracy"] = float(res["accuracy"])
                    if "accuracy_norm" in res:
                        log_metrics["eval/piqa_accuracy_norm"] = float(res["accuracy_norm"])

                elif t == "openbookqa":
                    from evals.tasks.openbookqa import run as run_openbookqa

                    res = run_openbookqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="test",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/openbookqa_acc_norm"] = float(res["acc_norm"])
                    log_metrics["e/openbookqa/acc_norm"] = float(res["acc_norm"])

                elif t == "commonsense_qa":
                    from evals.tasks.commonsense_qa import run as run_commonsense_qa

                    res = run_commonsense_qa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/commonsense_qa_acc_norm"] = float(res["acc_norm"])
                    log_metrics["e/commonsense_qa/acc_norm"] = float(res["acc_norm"])

                elif t == "siqa":
                    from evals.tasks.siqa import run as run_siqa

                    res = run_siqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/siqa_acc_norm"] = float(res["acc_norm"])
                    log_metrics["e/siqa/acc_norm"] = float(res["acc_norm"])

                elif t == "winogrande":
                    from evals.tasks.winogrande import run as run_winogrande

                    res = run_winogrande(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        config="winogrande_xl",
                        split="validation",
                        limit=eval_limit,
                    )
                    all_results.append(res)
                    log_metrics["eval/winogrande_accuracy"] = float(res["accuracy"])
                    log_metrics["e/winogrande/acc_norm"] = float(res.get("acc_norm", res["accuracy"]))

                elif t == "fineweb":
                    # FineWeb-v1/FineWeb-Edu blogpost aggregate score: mean of specific acc_norm columns.
                    from evals.tasks.arc import run as run_arc
                    from evals.tasks.commonsense_qa import run as run_commonsense_qa
                    from evals.tasks.hellaswag import run as run_hellaswag
                    from evals.tasks.mmlu_fineweb import run as run_mmlu_fineweb
                    from evals.tasks.openbookqa import run as run_openbookqa
                    from evals.tasks.piqa import run as run_piqa
                    from evals.tasks.siqa import run as run_siqa
                    from evals.tasks.winogrande import run as run_winogrande

                    inner_results = []
                    metrics = {}

                    # FineWeb blog ran lighteval with --max_samples 1000.
                    bench_limit = eval_limit
                    if bench_limit is None:
                        bench_limit = 1000

                    res_csqa = run_commonsense_qa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=bench_limit,
                    )
                    inner_results.append(res_csqa)
                    metrics["commonsense_qa/acc_norm"] = float(res_csqa["acc_norm"])

                    res_hs = run_hellaswag(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=bench_limit,
                    )
                    inner_results.append(res_hs)
                    metrics["hellaswag/acc_norm"] = float(res_hs["acc_norm"])

                    res_obqa = run_openbookqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="test",
                        limit=bench_limit,
                    )
                    inner_results.append(res_obqa)
                    metrics["openbookqa/acc_norm"] = float(res_obqa["acc_norm"])

                    res_piqa = run_piqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=bench_limit,
                    )
                    inner_results.append(res_piqa)
                    metrics["piqa/acc_norm"] = float(res_piqa["acc_norm"])

                    res_siqa = run_siqa(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        split="validation",
                        limit=bench_limit,
                    )
                    inner_results.append(res_siqa)
                    metrics["siqa/acc_norm"] = float(res_siqa["acc_norm"])

                    res_wg = run_winogrande(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        config="winogrande_xl",
                        split="validation",
                        limit=bench_limit,
                    )
                    inner_results.append(res_wg)
                    metrics["winogrande/acc_norm"] = float(res_wg["acc_norm"])

                    res_arc = run_arc(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        challenge=True,
                        split="test",
                        limit=bench_limit,
                    )
                    inner_results.append(res_arc)
                    metrics["arc/acc_norm"] = float(res_arc["acc_norm"])

                    res_mmlu = run_mmlu_fineweb(
                        model=model,
                        device=device,
                        autocast_dtype=autocast_dtype,
                        max_seq_len=max_seq_len,
                        subjects=str(getattr(self.config, "eval_mmlu_subjects", "all")),
                        limit=bench_limit,
                    )
                    inner_results.append(res_mmlu)
                    metrics["mmlu/acc_norm"] = float(res_mmlu["overall"]["acc_norm"])

                    required = [
                        "commonsense_qa/acc_norm",
                        "hellaswag/acc_norm",
                        "openbookqa/acc_norm",
                        "piqa/acc_norm",
                        "siqa/acc_norm",
                        "winogrande/acc_norm",
                        "arc/acc_norm",
                        "mmlu/acc_norm",
                    ]
                    agg_score = sum(metrics[k] for k in required) / len(required)

                    res = {
                        "task": "fineweb",
                        "agg_score": float(agg_score),
                        "metrics": metrics,
                        "required_metrics": required,
                        "results": inner_results,
                    }
                    all_results.append(res)

                    log_metrics["e/agg_score"] = float(agg_score)
                    for k, v in metrics.items():
                        log_metrics[f"e/{k}"] = float(v)

                else:
                    raise ValueError(f"Unknown eval task: {t}")
            except Exception as e:
                print(f"[eval_after_train] task={t} failed: {e}")
                all_results.append({"task": str(t), "error": str(e)})
                # numeric marker so it can be logged to tracking tools
                log_metrics[f"eval/{t}_failed"] = 1.0

        if out_dir:
            payload = {
                "step": int(step),
                "tasks": tasks,
                "results": all_results,
            }
            out_path = os.path.join(out_dir, f"eval_step{int(step):06d}.json")
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
            latest_path = os.path.join(out_dir, "latest.json")
            with open(latest_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")

            print(f"[eval_after_train] wrote: {out_path}")
            print(f"[eval_after_train] latest: {latest_path}")

            # Also record paths into the main run logfile for easy discovery.
            try:
                if getattr(self.logger, "logfile", None):
                    with open(self.logger.logfile, "a") as f:
                        f.write(f"eval_results_json:{out_path}\n")
                        f.write(f"eval_results_latest:{latest_path}\n")
            except Exception:
                pass

        if log_metrics:
            # Log evaluation metrics at the end of training.
            self.logger.log_step(int(step), log_metrics, prefix="eval")