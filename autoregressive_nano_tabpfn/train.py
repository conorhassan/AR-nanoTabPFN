"""Training loop for autoregressive-nanoTabPFN."""

import time
from typing import Optional, Iterator, Dict, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.optim import AdamW

from .model import ARNanoTabPFN


def train(
    model: ARNanoTabPFN,
    data_iter: Iterator[Dict[str, Any]],
    steps: int,
    lr: float = 1e-3,
    device: str = 'cuda',
    compile_model: bool = True,
    log_interval: int = 25,
) -> ARNanoTabPFN:
    """
    Train autoregressive-nanoTabPFN.

    Args:
        model: ARNanoTabPFN instance
        data_iter: Iterator yielding dicts with 'x', 'y', 'train_test_split_index'
        steps: Number of training steps
        lr: Learning rate
        device: Device to train on
        compile_model: Whether to use torch.compile
        log_interval: Steps between logging

    Returns:
        Trained model
    """
    model = model.to(device)

    if compile_model and device == 'cuda':
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = AdamW(model.parameters(), lr=lr)
    model.train()

    total_time = 0.0
    total_loss = 0.0

    for step in range(steps):
        t0 = time.perf_counter()

        batch = next(data_iter)
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        train_split = batch['train_test_split_index']

        # Forward with AMP
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = model(x, y, train_split)
            targets = y[:, train_split:].reshape(-1).long()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        step_time = time.perf_counter() - t0
        total_time += step_time
        total_loss += loss.item()

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            print(f"step {step+1:5d} | loss {avg_loss:.4f} | time {total_time:.1f}s")
            total_loss = 0.0

    return model


class DataIterator:
    """Simple iterator wrapper for HDF5 data compatible with nanoTabPFN format."""

    def __init__(self, filename: str, batch_size: int = 32, device: str = 'cuda'):
        import h5py
        self.filename = filename
        self.batch_size = batch_size
        self.device = device
        self.pointer = 0

        with h5py.File(filename, 'r') as f:
            self.num_samples = f['X'].shape[0]

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, Any]:
        import h5py

        with h5py.File(self.filename, 'r') as f:
            end = self.pointer + self.batch_size

            if end > self.num_samples:
                self.pointer = 0
                end = self.batch_size

            num_features = f['num_features'][self.pointer:end].max()
            num_datapoints = f['num_datapoints'][self.pointer:end].max()

            x = torch.from_numpy(f['X'][self.pointer:end, :num_datapoints, :num_features])
            y = torch.from_numpy(f['y'][self.pointer:end, :num_datapoints])
            train_split = f['single_eval_pos'][self.pointer]

            self.pointer = end

        return {
            'x': x.to(self.device),
            'y': y.to(self.device),
            'train_test_split_index': int(train_split),
        }


if __name__ == '__main__':
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ARNanoTabPFN(
        d_model=96,
        n_heads=4,
        n_layers=3,
        d_ff=192,
        num_classes=2
    )

    # Use nanoTabPFN's data format
    data_iter = DataIterator('300k_150x5_2.h5', batch_size=32, device=device)

    trained_model = train(
        model,
        data_iter,
        steps=2500,
        lr=4e-3,
        device=device,
        log_interval=25
    )
