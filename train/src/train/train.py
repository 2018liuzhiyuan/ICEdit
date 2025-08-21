from torch_npu.contrib import transfer_to_npu
import os
from torch.utils.data import DataLoader
import torch, torch_npu
import lightning as L
import yaml
import os
import random
import time
import numpy as np
from datasets import load_dataset

from .data import (
    EditDataset,
    OminiDataset,
    EditDataset_with_Omini
)
from .model import OminiModel
from .callbacks import TrainingCallback


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank


def get_config():
    config_path = os.environ.get("XFL_CONFIG")
    assert config_path is not None, "Please set the XFL_CONFIG environment variable"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def init_wandb(wandb_config, run_name):
    import wandb

    try:
        assert os.environ.get("WANDB_API_KEY") is not None
        wandb.init(
            project=wandb_config["project"],
            name=run_name,
            config={},
        )
    except Exception as e:
        print("Failed to initialize WanDB:", e)

import torch.distributed as dist
def main():
    # Initialize
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch_npu.npu.init()

    if world_size > 1:
        dist.init_process_group(backend='hccl', init_method='env://')
        torch_npu.npu.set_device(rank) 
        is_main_process = rank == 0
    else:
        torch_npu.npu.set_device(0)
        is_main_process = True
        
    
    config = get_config()
    training_config = config["train"]
    run_name = time.strftime("%Y%m%d-%H%M%S")
    
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    random.seed(seed)

    # Initialize WanDB
    wandb_config = training_config.get("wandb", None)
    if wandb_config is not None and is_main_process:
        init_wandb(wandb_config, run_name)

    print("Rank:", rank)
    if is_main_process:
        print("Config:", config)
        
    if 'use_offset_noise' not in config.keys():
        config['use_offset_noise'] = False

    # Initialize dataset and dataloader
    CACHE_DIR = '/dev/shm/dataset_cache' # 缓存路径
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if training_config["dataset"]["type"] == "edit":
        dataset = load_dataset('osunlp/MagicBrush', cache_dir=CACHE_DIR)
        dataset = EditDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
    elif training_config["dataset"]["type"] == "omini":
        dataset = load_dataset(training_config["dataset"]["path"], cache_dir=CACHE_DIR)
        dataset = OminiDataset(
            dataset,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )

    elif training_config["dataset"]["type"] == "edit_with_omini":
        omni = load_dataset("parquet", data_files=os.path.abspath(training_config["dataset"]["path"]), split="train", cache_dir=CACHE_DIR)
        magic = load_dataset('osunlp/MagicBrush', cache_dir=CACHE_DIR)
        dataset = EditDataset_with_Omini(
            magic,
            omni,
            condition_size=training_config["dataset"]["condition_size"],
            target_size=training_config["dataset"]["target_size"],
            drop_text_prob=training_config["dataset"]["drop_text_prob"],
        )
          

    print("Dataset length:", len(dataset))
    train_loader = DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config["dataloader_workers"],
        pin_memory=True,  
    )

    # Initialize model
    trainable_model = OminiModel(
        flux_fill_id=config["flux_path"],
        lora_config=training_config["lora_config"],
        device=f"npu",
        dtype=getattr(torch, config["dtype"]),
        optimizer_config=training_config["optimizer"],
        model_config=config.get("model", {}),
        gradient_checkpointing=training_config.get("gradient_checkpointing", False),
        use_offset_noise=config["use_offset_noise"],
    )

    # Callbacks for logging and saving checkpoints
    training_callbacks = (
        [TrainingCallback(run_name, training_config=training_config)]
        if is_main_process
        else []
    )
    # Save config
    save_path = training_config.get("save_path", "./output")
    if is_main_process:
        os.makedirs(f"{save_path}/{run_name}")
        with open(f"{save_path}/{run_name}/config.yaml", "w") as f:
            yaml.dump(config, f)

    # Initialize trainer
    from lightning.pytorch.accelerators import Accelerator
    class NPUAccelerator(Accelerator):
        @staticmethod
        def is_available() -> bool:
            return torch_npu.npu.is_available()

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device(f"npu:{i}") for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            return torch_npu.npu.device_count()
        
    NPUAccelerator.register_accelerators(NPUAccelerator)
    
    trainer = L.Trainer(
        accelerator='npu',
        devices=world_size,
        strategy='ddp' if world_size > 1 else 'auto',
        accumulate_grad_batches=training_config["accumulate_grad_batches"],
        callbacks=training_callbacks,
        enable_checkpointing=False,
        default_root_dir=f"{save_path}/{run_name}",
        enable_progress_bar=True,
        logger=False,
        max_steps=training_config.get("max_steps", -1),
        max_epochs=training_config.get("max_epochs", -1),
        gradient_clip_val=training_config.get("gradient_clip_val", 0.5),
    )

    setattr(trainer, "training_config", training_config)

    
    # Start training
    trainer.fit(trainable_model, train_loader)


if __name__ == "__main__":
    main()
