"""
Utility functions for logging, wandb integration, and other helper functions.
"""
import os
import uuid
import time
import subprocess
from typing import Optional, Any

import torch

# Import wandb with fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging will be limited to console/file")

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("Warning: swanlab not available, logging will be limited to console/file")



class Logger:
    """
    A logging utility that handles file logging and wandb integration.
    """
    
    def __init__(self, config, master_process: bool = False):
        """
        Initialize the logger.
        
        Args:
            config: Training configuration
            master_process: Whether this is the master process for logging
        """
        self.config = config
        self.master_process = master_process
        self.run_id = None
        self.logdir = None
        self.logfile = None
        self.wandb_initialized = False
        self.swanlab_initialized = False
        if self.master_process:
            self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging directories and files."""
        self.run_id = str(uuid.uuid4())
        self.logdir = f'logs/{self.run_id}/'
        os.makedirs(self.logdir, exist_ok=True)
        self.logfile = f'logs/{self.run_id}.txt'
        
        # Create the log file with initial information
        with open(self.logfile, "w") as f:
            f.write('=' * 100 + '\n')
            # Log the training script code
            try:
                import sys
                with open(sys.argv[0]) as code_file:
                    code = code_file.read()
                f.write(code)
            except Exception as e:
                f.write(f"Could not log training script: {e}\n")
            f.write('=' * 100 + '\n')
            
            # Log hardware/software environment
            f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
            try:
                result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                f.write(f'{result.stdout}\n')
            except Exception as e:
                f.write(f"Could not run nvidia-smi: {e}\n")
            f.write('=' * 100 + '\n')
    
    def init_wandb(self):
        """Initialize wandb logging if available and requested."""
        if not self.master_process:
            return
            
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb_run_name = self.config.wandb_run_name if self.config.wandb_run_name else str(self.run_id)
            wandb.init(
                project=self.config.wandb_project,
                name=wandb_run_name,
                config=vars(self.config)  # log all arguments
            )
            self.wandb_initialized = True
            print(f"Initialized wandb project: {self.config.wandb_project}, run: {wandb_run_name}")
        elif self.config.use_wandb and not WANDB_AVAILABLE:
            print("Warning: wandb requested but not available")
    
    def init_swanlab(self):
        """Initialize swanlab logging if available and requested."""
        if not self.master_process:
            return
        
        if self.config.use_swanlab and SWANLAB_AVAILABLE:
            swanlab_run_name = self.config.swanlab_run_name if self.config.swanlab_run_name else str(self.run_id)
            swanlab.init(project=self.config.swanlab_project, name=swanlab_run_name, config=vars(self.config))
            self.swanlab_initialized = True
            print(f"Initialized swanlab project: {self.config.swanlab_project}, run: {swanlab_run_name}")
        elif self.config.use_swanlab and not SWANLAB_AVAILABLE:
            print("Warning: swanlab requested but not available")
    
    def log_step(self, step: int, metrics: dict, prefix: str = ""):
        """
        Log metrics for a training step.
        
        Args:
            step: Current step number
            metrics: Dictionary of metrics to log
            prefix: Prefix for log messages (e.g., "train", "val")
        """
        if not self.master_process:
            return
        
        # Create log message
        log_parts = [f"step:{step}"]
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            log_parts.append(f"{key}:{value:.4f}" if isinstance(value, float) else f"{key}:{value}")
        
        log_message = f"{prefix} " + " ".join(log_parts) if prefix else " ".join(log_parts)
        
        # Log to console
        print(log_message)
        
        # Log to file
        if self.logfile:
            with open(self.logfile, "a") as f:
                f.write(log_message + '\n')
        
        # Log to wandb
        if self.wandb_initialized:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    wandb_metrics[key] = value.item()
                else:
                    wandb_metrics[key] = value
            wandb_metrics["step"] = step
            wandb.log(wandb_metrics, step=step)

        if self.swanlab_initialized:
            swanlab_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor):
                    swanlab_metrics[key] = value.item()
                else:
                    swanlab_metrics[key] = value
            swanlab_metrics["step"] = step
            swanlab.log(swanlab_metrics, step=step)
    
    def save_checkpoint(
        self,
        step: int,
        code: str,
        model_state: dict,
        optimizer_states: list,
        *,
        training_config: Optional[dict] = None,
        model_config: Optional[dict] = None,
    ):
        """
        Save a training checkpoint.
        
        Args:
            step: Current step number
            code: Training script code
            model_state: Model state dictionary
            optimizer_states: List of optimizer state dictionaries
            training_config: (Optional) Serializable training config dict
            model_config: (Optional) Serializable model config dict (vocab_size, n_layer, ...)
        """
        if not self.master_process or not self.run_id:
            return
        
        log = {
            'step': step,
            'code': code,
            'model': model_state,
            'optimizers': optimizer_states
        }
        if training_config is not None:
            log["training_config"] = training_config
        if model_config is not None:
            log["model_config"] = model_config
        checkpoint_path = f'logs/{self.run_id}/state_step{step:06d}.pt'
        torch.save(log, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def finish(self):
        """Finish logging and clean up resources."""
        if self.master_process and self.wandb_initialized:
            wandb.finish()
        if self.master_process and self.swanlab_initialized:
            swanlab.finish()

def get_memory_usage():
    """Get current GPU memory usage in MiB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() // 1024 // 1024
    return 0


def calculate_steps(config, ddp_world_size: int):
    """
    Calculate training and validation steps based on configuration.
    
    Args:
        config: Training configuration
        ddp_world_size: Total number of processes
        
    Returns:
        Tuple of (val_steps, train_accumulation_steps)
    """
    B, T = config.device_batch_size, config.sequence_length
    
    # Calculate validation steps
    assert config.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = config.val_tokens // (B * T * ddp_world_size)
    
    # Calculate gradient accumulation steps
    assert config.batch_size % (B * ddp_world_size) == 0
    train_accumulation_steps = config.batch_size // (B * ddp_world_size)
    
    return val_steps, train_accumulation_steps


def setup_distributed():
    """
    Set up distributed training environment.
    
    Returns:
        Tuple of (ddp_rank, ddp_local_rank, ddp_world_size, device, master_process)
    """
    import torch.distributed as dist
    
    assert torch.cuda.is_available()
    dist.init_process_group(backend='nccl')
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    
    print(f"using device: {device}")
    
    return ddp_rank, ddp_local_rank, ddp_world_size, device, master_process