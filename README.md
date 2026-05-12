# fair-prep — research

LoRA bench for small language models + cross-size LoRA transfer + merging scaling-law experiments. Unified `lb` CLI.

Models supported: Qwen3 (0.6B → 14B)

## Sub-projects

| Path | What |
|------|------|
| [`lora-bench/`](lora-bench/) | SFT / DPO / OnPolicy-DPO-GTW + cross-size LoRA transfer + merging scaling-law (arXiv:2509.24244). Typer + Rich CLI. |

## Quick start

```bash
cd lora-bench
uv sync
./lb --help
./lb status            # snapshot runs/ + results/ + GPU
./lb models            # list model presets
```

Optional global alias:
```bash
echo 'alias lb="'"$(pwd)"'/lb"' >> ~/.zshrc && source ~/.zshrc
```

## Train

```bash
./lb train sft  -M qwen3-0.6b
./lb train dpo  -M qwen3-0.6b -k runs/qwen3_06b_sft_gsm8k/final
./lb train gtw  -M qwen3-0.6b -k runs/qwen3_06b_sft_gsm8k/final
```

`-M` accepts a preset alias (see `lb models`) or any HF id (`-M Qwen/Qwen3-32B`).

## Evaluate

```bash
./lb eval gsm8k   -a runs/.../final -n 200 -o results/sft.json
./lb eval math500 -a runs/.../final -n 500 -o results/math500_sft.json
./lb report                                  # build results/REPORT.md
./lb analyze results/math500_*.json          # categorize failures
```

## Cross-size transfer (research axis)

Inspired by **arXiv:2509.24244** (Wang et al, ICML 2026 — "Model Merging Scaling Laws"). The paper merges experts on the same base; this repo pushes the idea cross-size: train LoRA experts on **Qwen3-4B teacher**, project to **Qwen3-0.6B student** hidden space via OLS on calibration activations, then merge `k` transferred adapters and check whether `L(k) = L∞ + A/(k+b)` still holds on the student.

```bash
# 1. train teacher per math domain
./lb train sft -M qwen3-4b -c configs/teacher_4b_algebra.yaml

# 2. project teacher LoRA → student
./lb transfer \
    -T Qwen/Qwen3-4B-Instruct-2507 \
    -S Qwen/Qwen3-0.6B \
    -a runs/qwen3_4b_teacher_algebra/final \
    -C data/calib_math.jsonl \
    -o runs/transferred/algebra/adapter

# 3. merge k transferred adapters
./lb merge -m ties \
    -a runs/transferred/algebra \
    -a runs/transferred/geometry \
    -o runs/merged_ties_k2

# 4. eval + fit scaling law
./lb eval math500 -a runs/merged_ties_k2 -o results/scaling/ties_k2.json
./lb fit k -p results/scaling/ties_pairs.json -o results/scaling/ties_fit.json
```

Falsifiable hypothesis: the law `L∞ + A/(k+b)` is preserved under cross-size transfer (student-dependent `L∞`), with a residual gap `Δ_transfer` vs natively-trained 0.6B experts.

## Layout

```
fair-prep/
├── README.md
├── pyproject.toml      # deps (torch, transformers, peft, trl, typer, rich, scipy, sklearn, …)
└── lora-bench/
    ├── lb              # CLI launcher
    ├── configs/        # base + per-task + teacher per-domain YAML
    ├── scripts/        # 01..09 baseline pipeline, 20..22 transfer pipeline
    ├── src/
    │   ├── cli/app.py  # Typer + Rich unified CLI
    │   ├── core/       # model loading, tokenizer, hooks, config, utils
    │   ├── data/       # gsm8k / math500 / preferences / math_subset
    │   ├── trainers/   # sft, dpo, op_dpo_gtw, replay_buffer
    │   ├── eval/       # runner + per-dataset scorers + analyze
    │   ├── transfer/   # cross-size LoRA projection + CKA layer align
    │   ├── merge/      # average / TA / TIES / DARE
    │   ├── rewards/    # GSM8K reward + registry
    │   ├── scaling_law.py
    │   └── report.py
    ├── runs/           # checkpoints (gitignored)
    └── results/        # eval JSONs + REPORT.md
```

## Research components

### OnPolicy-DPO-GTW
`src/trainers/op_dpo_gtw.py` — combines online generation, verifier-based pairing, Group-LOO token-level credit. Free process supervision without training a PRM. Loss = `α₁·DPO_sigmoid + α₂·DPO_token_weighted + α₃·SFT_chosen + α₄·KL`.

### Cross-size LoRA transfer
`src/transfer/cross_lora.py` — collects activations on both teacher + student, solves ridge-OLS for input/output projections, projects `A_s = A_t · pinv(Q_in^T)`, `B_s = Q_out^T · B_t`. `src/transfer/layer_align.py` provides CKA-based layer matching for differing depths.

### Merging scaling law
`src/scaling_law.py` — fits `L(k) = L∞ + A/(k+b)` (Theorem 3.1 of arXiv:2509.24244) and the size-dependent variant `L∞(N) = L* + B·N^-β`, `A(N) = A0·N^-γ`. Three-point fit suffices per the paper.

## Stack

- Python 3.11–3.12
- PyTorch ≥ 2.4 (CUDA preferred, MPS supported)
- Transformers + PEFT + TRL + Datasets
- Typer + Rich (CLI)
- SciPy + scikit-learn (OLS / CKA / curve fit)
- `uv` for dependency management

## License

Research code, MIT.
