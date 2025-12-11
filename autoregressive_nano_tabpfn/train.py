"""Training loop for ARTabPFN with online tabular data generation."""

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .model import ARTabPFN, create_dense_mask, create_row_mask
from .data import OnlineTabularDataset


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create cosine learning rate schedule with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train(
    model: ARTabPFN,
    dataset: OnlineTabularDataset,
    config: dict,
    device: str = "cuda",
) -> ARTabPFN:
    """Train ARTabPFN with online data generation."""
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    checkpoint_cfg = config.get("checkpoint", {})

    steps = training_cfg.get("max_steps", 20000)
    grad_clip = training_cfg.get("grad_clip", 1.0)
    compile_model = training_cfg.get("compile_model", True)
    use_amp = training_cfg.get("use_amp", True)
    log_interval = training_cfg.get("log_interval", 50)
    save_interval = checkpoint_cfg.get("save_interval", 1000)
    save_dir = checkpoint_cfg.get("save_dir", "checkpoints")

    model = model.to(device)

    if compile_model and device == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = AdamW(
        model.parameters(),
        lr=optimizer_cfg.get("lr", 1e-4),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        weight_decay=optimizer_cfg.get("weight_decay", 0.0),
    )

    scheduler = None
    if scheduler_cfg.get("use_scheduler", False):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=scheduler_cfg.get("warmup_steps", 2000),
            total_steps=scheduler_cfg.get("total_steps", steps),
        )

    model.train()

    loader = DataLoader(dataset, batch_size=None, shuffle=False)
    data_iter = iter(loader)

    mask_cache: Dict[tuple, tuple] = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    total_loss = 0.0
    t0 = time.perf_counter()

    for step in range(steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(device)

        Nc, Nb, Nt = batch.xc.size(1), batch.xb.size(1), batch.xt.size(1)
        cache_key = (Nc, Nb, Nt)

        if cache_key not in mask_cache:
            mask_features = create_dense_mask(seq_len=1, device=device)
            mask_rows = create_row_mask(
                num_rows=Nc + Nb + Nt,
                context_len=Nc,
                buffer_len=Nb,
                device=device,
            )
            mask_cache[cache_key] = (mask_features, mask_rows)
        else:
            mask_features, mask_rows = mask_cache[cache_key]

        with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp and device == "cuda"):
            loss = model(
                x_context=batch.xc,
                y_context=batch.yc.squeeze(-1),
                x_buffer=batch.xb,
                y_buffer=batch.yb.squeeze(-1),
                x_target=batch.xt,
                mask_features=mask_features,
                mask_rows=mask_rows,
                y_target=batch.yt.squeeze(-1),
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / log_interval
            steps_per_sec = log_interval / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {steps_per_sec:.1f} it/s")
            total_loss = 0.0
            t0 = time.perf_counter()

        if save_dir and save_interval > 0 and (step + 1) % save_interval == 0:
            ckpt_path = Path(save_dir) / f"step_{step+1}.pt"
            torch.save({"step": step + 1, "model": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    return model


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(config: Optional[dict] = None):
    if config is None:
        config = {}

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    dataset = OnlineTabularDataset(
        batch_size=data_cfg.get("batch_size", 512),
        num_batches=data_cfg.get("num_batches_per_epoch", 2000),
        d_list=data_cfg.get("d_list", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        nc_list=data_cfg.get("nc_list", [8, 16, 32, 64, 128, 256, 512, 1024]),
        num_buffer=data_cfg.get("num_buffer", 32),
        num_target=data_cfg.get("num_target", 512),
        normalize_y=data_cfg.get("normalize_y", True),
        dtype=getattr(torch, data_cfg.get("dtype", "float32")),
        device="cpu",
        seed=data_cfg.get("seed", 123),
    )

    model = ARTabPFN(
        d_model=model_cfg.get("d_model", 64),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 12),
        d_ff=model_cfg.get("d_ff", 128),
        num_features=model_cfg.get("num_features", 10),
        buffer_size=model_cfg.get("buffer_size", 32),
        num_components=model_cfg.get("num_components", 20),
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train(model, dataset, config, device=device)
    print("Training complete!")

    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else {}
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.batch_size is not None:
        config.setdefault("data", {})["batch_size"] = args.batch_size
    if args.device is not None:
        config["device"] = args.device

    main(config)
