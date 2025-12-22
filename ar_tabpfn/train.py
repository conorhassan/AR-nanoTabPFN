"""Training loop for ARTabPFN with online tabular data generation."""

import os
os.environ["TORCHINDUCTOR_DISABLE_CUDAGRAPHS"] = "1"
os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "0"

import torch._inductor.config
torch._inductor.config.triton.cudagraphs = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.persistent_reductions = True
if hasattr(torch._inductor.config.triton, "cudagraph_trees"):
    torch._inductor.config.triton.cudagraph_trees = False
if hasattr(torch._inductor.config, "use_static_cuda_launcher"):
    torch._inductor.config.use_static_cuda_launcher = False
if hasattr(torch._inductor.config, "use_static_triton_launcher"):
    torch._inductor.config.use_static_triton_launcher = False

import argparse
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .model import ARTabPFN, create_dense_mask, create_row_mask
from .data import OnlineTabularDataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class DataConfig:
    batch_size: int = 512
    num_batches_per_epoch: int = 2000
    num_workers: int = 0
    d_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    nc_list: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128, 256, 512, 1024])
    num_buffer: int = 32
    num_target: int = 512
    normalize_x: bool = True
    normalize_y: bool = True
    dtype: str = "float32"
    seed: int = 123


@dataclass
class ModelConfig:
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 12
    d_ff: int = 128
    num_features: int = 10
    buffer_size: int = 32
    num_components: int = 20


@dataclass
class TrainingConfig:
    max_steps: int = 20000
    grad_clip: float = 1.0
    compile_model: bool = True
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    val_interval: int = 250
    precompile_masks: bool = True
    precompile_shapes: Optional[List[List[int]]] = None  # [[Nc, Nb, Nt], ...]


@dataclass
class OptimizerConfig:
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0


@dataclass
class SchedulerConfig:
    use_scheduler: bool = False
    warmup_steps: int = 2000
    total_steps: Optional[int] = None


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    save_interval: int = 1000


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    project: str = "artabpfn"
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    log_interval: int = 50


@dataclass
class Config:
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            device=d.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            optimizer=OptimizerConfig(**{
                k: tuple(v) if k == "betas" else v
                for k, v in d.get("optimizer", {}).items()
            }),
            scheduler=SchedulerConfig(**d.get("scheduler", {})),
            checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )


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


def precompile_masks(
    shapes: List[Tuple[int, int, int]], device: str
) -> Dict[Tuple[int, int, int], Tuple]:
    """Precompile masks for given (Nc, Nb, Nt) shapes.

    Args:
        shapes: List of (context_len, buffer_len, target_len) tuples
        device: Device to create masks on

    Returns:
        Dictionary mapping (Nc, Nb, Nt) -> (mask_features, mask_rows)
    """
    mask_cache = {}
    print(f"Precompiling {len(shapes)} mask shapes...")

    for nc, nb, nt in shapes:
        key = (nc, nb, nt)
        num_rows = nc + nb + nt

        mask_features = create_dense_mask(seq_len=1, device=device)
        mask_rows = create_row_mask(
            num_rows=num_rows,
            context_len=nc,
            buffer_len=nb,
            device=device,
        )
        mask_cache[key] = (mask_features, mask_rows)
        print(f"  Precompiled: Nc={nc}, Nb={nb}, Nt={nt}")

    return mask_cache


class CudaPrefetcher:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device == "cuda" else None
        self.iter = iter(loader)
        self.next_batch = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            batch = next(self.iter)

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = batch.to(self.device, non_blocking=True)
        else:
            self.next_batch = batch.to(self.device)

    def __iter__(self):
        return self

    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self._preload()
        return batch


def train(model: ARTabPFN, dataset: OnlineTabularDataset, config: Config) -> ARTabPFN:
    """Train ARTabPFN with online data generation."""
    device = config.device
    model = model.to(device)

    # Initialize wandb if enabled
    use_wandb = config.logging.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=config.logging.project,
            name=config.logging.run_name,
            config=asdict(config),
            tags=config.logging.tags,
        )

    if config.training.compile_model and device == "cuda":
        model.embedder = torch.compile(model.embedder, dynamic=False)
        model.head = torch.compile(model.head, dynamic=False)

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        weight_decay=config.optimizer.weight_decay,
    )

    scheduler = None
    if config.scheduler.use_scheduler:
        total_steps = config.scheduler.total_steps or config.training.max_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=config.scheduler.warmup_steps,
            total_steps=total_steps,
        )

    model.train()

    loader = DataLoader(dataset, batch_size=None, shuffle=False, pin_memory=(device == "cuda"))
    prefetcher = CudaPrefetcher(loader, device)

    # Precompile masks if configured
    if config.training.precompile_masks and config.training.precompile_shapes:
        shapes = [tuple(s) for s in config.training.precompile_shapes]
        mask_cache = precompile_masks(shapes, device)
    else:
        mask_cache: Dict[tuple, tuple] = {}

    if config.checkpoint.save_dir:
        os.makedirs(config.checkpoint.save_dir, exist_ok=True)

    total_loss = 0.0
    t0 = time.perf_counter()

    for step in range(config.training.max_steps):
        batch = next(prefetcher)

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

        amp_dtype = getattr(torch, config.training.amp_dtype)
        with torch.autocast(
            device, dtype=amp_dtype, enabled=config.training.use_amp and device == "cuda"
        ):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if (step + 1) % config.logging.log_interval == 0:
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / config.logging.log_interval
            steps_per_sec = config.logging.log_interval / elapsed
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step {step+1:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | {steps_per_sec:.1f} it/s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/learning_rate": lr,
                        "train/steps_per_sec": steps_per_sec,
                    },
                    step=step + 1,
                )

            total_loss = 0.0
            t0 = time.perf_counter()

        if (
            config.checkpoint.save_dir
            and config.checkpoint.save_interval > 0
            and (step + 1) % config.checkpoint.save_interval == 0
        ):
            ckpt_path = Path(config.checkpoint.save_dir) / f"step_{step+1}.pt"
            torch.save({"step": step + 1, "model": model.state_dict()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Finalize wandb
    if use_wandb:
        wandb.summary["total_steps"] = config.training.max_steps
        wandb.summary["final_loss"] = avg_loss
        wandb.finish()

    return model


def load_config(config_path: str) -> Config:
    """Load YAML config file."""
    with open(config_path) as f:
        return Config.from_dict(yaml.safe_load(f) or {})


def print_system_info():
    """Print system/version info for debugging."""
    import torch
    print("=" * 60)
    print("System Info:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU capability: {torch.cuda.get_device_capability(0)}")
    try:
        import triton
        print(f"  Triton version: {triton.__version__}")
    except ImportError:
        print("  Triton: not installed")
    print("=" * 60)


def main(config: Optional[Config] = None):
    if config is None:
        config = Config()

    print_system_info()
    print(f"Using device: {config.device}")

    dataset = OnlineTabularDataset(
        batch_size=config.data.batch_size,
        num_batches=config.data.num_batches_per_epoch,
        d_list=config.data.d_list,
        nc_list=config.data.nc_list,
        num_buffer=config.data.num_buffer,
        num_target=config.data.num_target,
        normalize_x=config.data.normalize_x,
        normalize_y=config.data.normalize_y,
        dtype=getattr(torch, config.data.dtype),
        device="cpu",
        seed=config.data.seed,
    )

    model = ARTabPFN(
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        num_features=config.model.num_features,
        buffer_size=config.model.buffer_size,
        num_components=config.model.num_components,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train(model, dataset, config)
    print("Training complete!")

    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None, help="Override max training steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else Config()

    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.device is not None:
        config.device = args.device

    main(config)
