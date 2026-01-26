import torch
import numpy as np
import os
from torchvision.utils import make_grid, save_image
import logging

logger = logging.getLogger(__name__)

def check_data_leakage(train_dataset, val_dataset):
    """
    Check for data leakage between train and validation sets.
    Requires datasets to implement `get_sample_id(index)`.
    """
    if not hasattr(train_dataset, 'get_sample_id') or not hasattr(val_dataset, 'get_sample_id'):
        logger.warning("Datasets do not implement 'get_sample_id(index)'. Skipping data leakage check.")
        return

    logger.info("Checking for data leakage...")
    train_ids = set()
    for i in range(len(train_dataset)):
        train_ids.add(train_dataset.get_sample_id(i))

    leakage_count = 0
    for i in range(len(val_dataset)):
        val_id = val_dataset.get_sample_id(i)
        if val_id in train_ids:
            leakage_count += 1
            if leakage_count < 5:
                 logger.error(f"Data Leakage Detected! ID {val_id} found in both train and val.")

    if leakage_count > 0:
        raise RuntimeError(f"Data Leakage Detected! {leakage_count} samples overlap between train and val.")

    logger.info("Data leakage check passed.")

def visual_sanity_check(loader, output_dir, limit=1):
    """
    Save a batch of images to disk for visual inspection.
    """
    logger.info(f"Running visual sanity check... saving to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for i, batch in enumerate(loader):
        if i >= limit:
            break

        images = None
        if isinstance(batch, dict):
            if "image" in batch:
                images = batch["image"]
            elif "pixel_values" in batch:
                images = batch["pixel_values"]
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        elif isinstance(batch, torch.Tensor):
            images = batch

        if images is not None and isinstance(images, torch.Tensor):
            # Check shape
            if images.dim() == 4: # B, C, H, W
                try:
                    # Normalize to [0, 1] for visualization if needed, or just clamp
                    # Assuming images might be normalized with mean/std.
                    # We simply clamp and save.
                    # Or min-max normalize per image?
                    img_vis = images.clone().detach().cpu()
                    # Simple min-max scaling to 0-1
                    min_val = img_vis.min()
                    max_val = img_vis.max()
                    if max_val > min_val:
                        img_vis = (img_vis - min_val) / (max_val - min_val)

                    save_image(img_vis, os.path.join(output_dir, f"sanity_check_batch_{i}.png"))
                    logger.info(f"Saved sanity_check_batch_{i}.png")
                except Exception as e:
                    logger.warning(f"Failed to save visual sanity check image: {e}")
            else:
                logger.warning(f"Batch images have shape {images.shape}, expected 4D (B, C, H, W). Skipping visualization.")
        else:
             logger.warning("Could not extract images from batch for visual sanity check.")

def check_system_config(config):
    """
    Check system configuration for potential issues.
    """
    data_config = config.get("data", {}).get("data_loader_params", {})
    num_workers = data_config.get("num_workers", 0)

    if num_workers == 0:
        logger.warning("⚠️  num_workers is set to 0. This might be a bottleneck. Consider increasing it.")

    # Check pin_memory? DataLoader defaults.
    # We can check config if it exists.
    if "pin_memory" in data_config and not data_config["pin_memory"] and torch.cuda.is_available():
        logger.warning("⚠️  pin_memory is False but CUDA is available. Consider setting it to True.")

    # Check for magic numbers?
    # Hard to check programmatically without specific rules.
