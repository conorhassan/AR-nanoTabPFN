# AR-TabPFN

Efficient *autoregressive sampling* and *log-density evaluation* from TabPFN model.

## Install

```bash
pip install -e .
```

## flex_attention Compilation for Training

Key settings to get `flex_attention` working with `torch.compile`:

1. **Pre-compile flex_attention at module load time** (before model compilation):
   ```python
   from torch.nn.attention.flex_attention import flex_attention, create_block_mask

   if torch.cuda.is_available():
       flex_attention = torch.compile(flex_attention, fullgraph=True)
       create_block_mask = torch.compile(create_block_mask)
   ```

2. **Use `dynamic=False`** when compiling model/components:
   ```python
   model.embedder = torch.compile(model.embedder, dynamic=False)
   model.head = torch.compile(model.head, dynamic=False)
   ```

3. **Batch size limit**: Large batch sizes can cause Triton grid overflow errors. `batch_size=64` works reliably.

4. **Inductor settings** (set before imports):
   ```python
   torch._inductor.config.triton.cudagraphs = False
   torch._inductor.config.triton.unique_kernel_names = True
   torch._inductor.config.coordinate_descent_tuning = True
   ```

Compiling the whole model (`torch.compile(model)`) works but has long warmup time (~10+ min) due to tracing all shape combinations.

## Citation

```bibtex
@misc{hassan2025efficientautoregressiveinferencetransformer,
      title={Efficient Autoregressive Inference for Transformer Probabilistic Models},
      author={Conor Hassan and Nasrulloh Loka and Cen-You Li and Daolang Huang and Paul E. Chang and Yang Yang and Francesco Silvestrin and Samuel Kaski and Luigi Acerbi},
      year={2025},
      eprint={2510.09477},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2510.09477},
}
```
