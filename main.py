"""
Main training script for ACT Descent - Refactored modular version.

This script coordinates the training process using the modular components:
- args.py: Configuration and argument parsing
- data_loader.py: Data loading utilities
- optimizers.py: Optimizer setup and management
- trainer.py: Main training loop and logic
- utils.py: Logging and utility functions
"""
import os
import torch
import torch.distributed as dist

# Set wandb base URL before importing other modules
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'

from args import parse_args
from trainer import Trainer
from utils import setup_distributed


def main():
    """Main training function."""
    # Parse arguments
    config = parse_args()
    
    # Set up distributed training
    ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed()

    # Performance knobs (A100: enable TF32 TensorCore matmuls for FP32 ops)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    # Create and run trainer
    trainer = Trainer(
        config=config,
        ddp_rank=ddp_rank,
        ddp_local_rank=ddp_local_rank,
        ddp_world_size=ddp_world_size,
        device=device,
        master_process=master_process
    )
    
    # Run training
    trainer.train()
    # Finish logging (wandb/swanlab)
    trainer.finish()
    
    # Clean up distributed training
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()