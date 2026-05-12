"""`lb` — unified LoRA-bench CLI.

Subcommands:
    lb train sft|dpo|gtw    Train LoRA adapter
    lb eval gsm8k|math500   Evaluate adapter on benchmark
    lb infer                Single-prompt generation
    lb transfer             Cross-size LoRA projection (teacher → student)
    lb merge                Merge k LoRA adapters (avg | ta | ties | dare)
    lb fit                  Fit merging scaling law L = L∞ + A/(k+b)
    lb report               Build results/REPORT.md
    lb analyze              Categorize MATH-500 failures
    lb status               Show runs/, results/, GPU state
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.config import load_run_cfg
from ..core.utils import get_device, set_seed

console = Console()
err_console = Console(stderr=True, style="bold red")

app = typer.Typer(
    name="lb",
    help="LoRA-bench: SFT / DPO / GTW + cross-size transfer + merging scaling law.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
train_app = typer.Typer(help="Train LoRA adapter (SFT, DPO, OnPolicy-DPO-GTW).", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate adapter on a benchmark.", no_args_is_help=True)
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")

# ──────────────────────────── model presets ────────────────────────────

MODEL_PRESETS: dict[str, dict] = {
    # alias → {hf id, hidden, layers, family}
    "qwen3-0.6b":   {"hf": "Qwen/Qwen3-0.6B",            "hidden": 1024, "layers": 28, "family": "qwen3"},
    "qwen3-1.7b":   {"hf": "Qwen/Qwen3-1.7B",            "hidden": 2048, "layers": 28, "family": "qwen3"},
    "qwen3-4b":     {"hf": "Qwen/Qwen3-4B-Instruct-2507","hidden": 2560, "layers": 36, "family": "qwen3"},
    "qwen3-8b":     {"hf": "Qwen/Qwen3-8B",              "hidden": 4096, "layers": 36, "family": "qwen3"},
    "qwen3-14b":    {"hf": "Qwen/Qwen3-14B",             "hidden": 5120, "layers": 40, "family": "qwen3"},
    "qwen2.5-0.5b": {"hf": "Qwen/Qwen2.5-0.5B-Instruct", "hidden": 896,  "layers": 24, "family": "qwen2.5"},
    "qwen2.5-1.5b": {"hf": "Qwen/Qwen2.5-1.5B-Instruct", "hidden": 1536, "layers": 28, "family": "qwen2.5"},
    "qwen2.5-3b":   {"hf": "Qwen/Qwen2.5-3B-Instruct",   "hidden": 2048, "layers": 36, "family": "qwen2.5"},
    "qwen2.5-7b":   {"hf": "Qwen/Qwen2.5-7B-Instruct",   "hidden": 3584, "layers": 28, "family": "qwen2.5"},
    "llama3.2-1b":  {"hf": "meta-llama/Llama-3.2-1B-Instruct", "hidden": 2048, "layers": 16, "family": "llama"},
    "llama3.2-3b":  {"hf": "meta-llama/Llama-3.2-3B-Instruct", "hidden": 3072, "layers": 28, "family": "llama"},
    "gemma2-2b":    {"hf": "google/gemma-2-2b-it",       "hidden": 2304, "layers": 26, "family": "gemma"},
}


def _resolve_model(name: str) -> str:
    """Resolve alias to HF id; pass through if already an HF id."""
    key = name.lower()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]["hf"]
    return name


# ──────────────────────────── helpers ────────────────────────────

def _init(
    cfg_path: str | None,
    base_cfg: str,
    device: str | None,
    model: str | None = None,
    lora_r: int | None = None,
) -> tuple[str, dict]:
    dev = device or get_device()
    cfg = load_run_cfg(base_cfg, cfg_path) if cfg_path else load_run_cfg(base_cfg)
    if model:
        hf = _resolve_model(model)
        cfg["model"]["name"] = hf
        cfg["tokenizer"]["name"] = hf
    if lora_r is not None:
        cfg["lora"]["r"] = lora_r
        cfg["lora"]["alpha"] = lora_r * 2
    set_seed(cfg.get("seed", 42))
    console.print(Panel.fit(
        f"[cyan]device[/]  {dev}\n"
        f"[cyan]model[/]   {cfg['model']['name']}\n"
        f"[cyan]cfg[/]     {cfg_path or '(base only)'}\n"
        f"[cyan]lora_r[/]  {cfg['lora']['r']}\n"
        f"[cyan]run[/]     {cfg.get('run_name', '—')}",
        title="lb", border_style="cyan",
    ))
    return dev, cfg


def _metrics_table(metrics: dict, title: str) -> Table:
    t = Table(title=title, show_header=False, border_style="green")
    t.add_column("metric", style="cyan")
    t.add_column("value")
    for k in ("accuracy", "n", "tokens_per_sec", "wall_seconds", "peak_memory_mb"):
        if k in metrics and metrics[k] is not None:
            v = metrics[k]
            t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    return t


# ──────────────────────────── train ────────────────────────────

@train_app.command("sft")
def train_sft(
    cfg: str = typer.Option("configs/sft_gsm8k.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M",
                                        help="HF id or preset (lb models)"),
    lora_r: Optional[int] = typer.Option(None, "--lora-r", help="override LoRA rank"),
    max_steps: int = typer.Option(-1, "--max-steps", "-s"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples", "-n"),
    fast: bool = typer.Option(False, "--fast",
                              help="use unsloth FastLanguageModel (CUDA, 2-4× faster)"),
    no_4bit: bool = typer.Option(False, "--no-4bit",
                                 help="(unsloth only) disable 4-bit quant"),
):
    """Supervised fine-tuning."""
    dev, c = _init(cfg, base_cfg, device, model=model, lora_r=lora_r)
    if fast:
        try:
            from ..trainers import sft_unsloth
            sft_unsloth.run(c, dev, max_steps=max_steps,
                            max_train_samples=max_train_samples,
                            load_in_4bit=not no_4bit)
            return
        except (ImportError, RuntimeError) as e:
            err_console.print(f"[--fast] unsloth unavailable: {e}\nfalling back to TRL SFTTrainer.")
    from ..trainers import sft as sft_trainer
    sft_trainer.run(c, dev, max_steps=max_steps, max_train_samples=max_train_samples)


@train_app.command("dpo")
def train_dpo(
    cfg: str = typer.Option("configs/dpo_pref.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M"),
    sft_checkpoint: Optional[str] = typer.Option(None, "--sft-checkpoint", "-k"),
    max_steps: int = typer.Option(-1, "--max-steps", "-s"),
    max_train_samples: Optional[int] = typer.Option(None, "--max-train-samples", "-n"),
):
    """Direct preference optimization (standard TRL)."""
    from ..trainers import dpo as dpo_trainer
    dev, c = _init(cfg, base_cfg, device, model=model)
    ckpt = sft_checkpoint or c.get("sft_checkpoint")
    if not ckpt:
        err_console.print("need --sft-checkpoint or cfg.sft_checkpoint")
        raise typer.Exit(1)
    dpo_trainer.run(c, dev, sft_checkpoint=ckpt, max_steps=max_steps, max_train_samples=max_train_samples)


@train_app.command("gtw")
def train_gtw(
    cfg: str = typer.Option("configs/op_dpo_gtw.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M"),
    sft_checkpoint: Optional[str] = typer.Option(None, "--sft-checkpoint", "-k"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", "-s"),
    max_train_prompts: Optional[int] = typer.Option(None, "--max-train-prompts", "-n"),
):
    """OnPolicy-DPO-GTW (research, group-token-weighted)."""
    from ..trainers import op_dpo_gtw as gtw_trainer
    dev, c = _init(cfg, base_cfg, device, model=model)
    ckpt = sft_checkpoint or c.get("sft_checkpoint")
    if not ckpt:
        err_console.print("need --sft-checkpoint or cfg.sft_checkpoint")
        raise typer.Exit(1)
    gtw_trainer.run(c, dev, sft_checkpoint=ckpt, max_steps=max_steps, max_train_prompts=max_train_prompts)


# ──────────────────────────── eval ────────────────────────────

def _run_eval(dataset_name, cfg, base_cfg, device, adapter, max_samples, max_new_tokens, out,
              model=None, batch_size=8):
    from ..core.model import load_base_model, load_model_with_lora, load_tokenizer
    from ..core.utils import ensure_dir, log_env, save_json
    from ..eval import gsm8k as e_gsm8k, math500 as e_math500
    from ..eval.runner import run_eval

    ds_mod = {"gsm8k": e_gsm8k, "math500": e_math500}[dataset_name]
    dev, c = _init(cfg, base_cfg, device, model=model)
    tok = load_tokenizer(c)
    model = (load_model_with_lora(c, dev, adapter_path=adapter)
             if adapter else load_base_model(c, dev))
    eval_ds = ds_mod.load_dataset(c, tok, max_samples=max_samples)
    console.print(f"[dim]n_samples = {len(eval_ds)}  batch_size = {batch_size}[/]")
    metrics = run_eval(model, tok, eval_ds, dev, score_fn=ds_mod.score,
                      max_new_tokens=max_new_tokens, batch_size=batch_size)
    metrics["env"] = log_env(dev)
    metrics["adapter"] = adapter
    metrics["task"] = dataset_name
    console.print(_metrics_table(metrics, f"eval • {dataset_name}"))
    if out:
        ensure_dir(Path(out).parent)
        save_json(metrics, out)
        console.print(f"[dim]saved →[/] [green]{out}[/]")


@eval_app.command("gsm8k")
def eval_gsm8k(
    cfg: str = typer.Option("configs/sft_gsm8k.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    batch_size: int = typer.Option(8, "--batch-size", "-B"),
    out: Optional[str] = typer.Option(None, "--out", "-o"),
):
    """GSM8K (greedy + exact-match)."""
    _run_eval("gsm8k", cfg, base_cfg, device, adapter, max_samples, max_new_tokens, out,
              model, batch_size)


@eval_app.command("math500")
def eval_math500(
    cfg: str = typer.Option("configs/sft_gsm8k.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a"),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    batch_size: int = typer.Option(8, "--batch-size", "-B"),
    out: Optional[str] = typer.Option(None, "--out", "-o"),
):
    """MATH-500 (greedy + math_verify on boxed answer)."""
    _run_eval("math500", cfg, base_cfg, device, adapter, max_samples, max_new_tokens, out,
              model, batch_size)


# ──────────────────────────── infer ────────────────────────────

@app.command("infer")
def infer(
    question: str = typer.Argument(..., help="prompt text"),
    cfg: str = typer.Option("configs/sft_gsm8k.yaml", "--cfg", "-c"),
    base_cfg: str = typer.Option("configs/base.yaml", "--base-cfg"),
    device: Optional[str] = typer.Option(None, "--device", "-d"),
    model: Optional[str] = typer.Option(None, "--model", "-M"),
    adapter: Optional[str] = typer.Option(None, "--adapter", "-a"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    sample: bool = typer.Option(False, "--sample/--greedy"),
    temperature: float = typer.Option(0.7, "--temperature", "-t"),
):
    """Single-prompt inference."""
    import torch

    from ..core.model import load_base_model, load_model_with_lora, load_tokenizer
    from ..data import format_qwen_prompt_only

    dev, c = _init(cfg, base_cfg, device, model=model)
    tok = load_tokenizer(c)
    model = (load_model_with_lora(c, dev, adapter_path=adapter)
             if adapter else load_base_model(c, dev))
    model.eval()
    prompt = format_qwen_prompt_only(question, tok)
    with torch.no_grad():
        inputs = tok(prompt, return_tensors="pt").to(dev)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=sample, temperature=temperature,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    console.print(Panel(text, title="output", border_style="green"))


# ──────────────────────────── transfer ────────────────────────────

@app.command("transfer")
def transfer(
    teacher: str = typer.Option(..., "--teacher", "-T"),
    student: str = typer.Option(..., "--student", "-S"),
    adapter: str = typer.Option(..., "--adapter", "-a", help="teacher LoRA adapter dir"),
    calib: str = typer.Option(..., "--calib", "-C", help="jsonl prompts for activation calib"),
    out: str = typer.Option(..., "--out", "-o"),
    device: str = typer.Option("cuda", "--device", "-d"),
    n_calib: int = typer.Option(256, "--n-calib", "-n"),
    ridge: float = typer.Option(1e-4, "--ridge"),
):
    """Project teacher LoRA into student hidden space (OLS + recon error)."""
    from ..transfer.cross_lora import transfer_adapter
    errs = transfer_adapter(teacher, student, adapter, calib, out,
                            device=device, n_calib=n_calib, ridge=ridge)
    mean = sum(errs.values()) / max(len(errs), 1)
    t = Table(title="transfer recon error", border_style="green")
    t.add_column("module", style="cyan")
    t.add_column("rel err", justify="right")
    for k, v in sorted(errs.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        t.add_row(k, f"{v:.4f}")
    console.print(t)
    console.print(f"[cyan]mean err[/] = [bold]{mean:.4f}[/]  ({len(errs)} modules)")


# ──────────────────────────── merge ────────────────────────────

@app.command("merge")
def merge(
    method: str = typer.Option("average", "--method", "-m",
                               help="average | ta | ties | dare"),
    adapters: list[str] = typer.Option(..., "--adapter", "-a"),
    out: str = typer.Option(..., "--out", "-o"),
    top_frac: float = typer.Option(0.2, "--top-frac"),
    dare_p: float = typer.Option(0.5, "--dare-p"),
    seed: int = typer.Option(0, "--seed"),
):
    """Merge k LoRA adapters into one."""
    from ..merge.merge_lora import merge_adapters
    if method not in {"average", "ta", "ties", "dare"}:
        err_console.print(f"unknown method: {method}")
        raise typer.Exit(1)
    kw: dict = {}
    if method == "ties":
        kw["top_frac"] = top_frac
    elif method == "dare":
        kw["p"] = dare_p
        kw["seed"] = seed
    merge_adapters(adapters, out, method=method, **kw)
    console.print(f"[green]merged[/] k=[bold]{len(adapters)}[/] via [cyan]{method}[/] → {out}")


# ──────────────────────────── fit-law ────────────────────────────

fit_app = typer.Typer(help="Fit merging scaling law.", no_args_is_help=True)
app.add_typer(fit_app, name="fit")


@fit_app.command("k")
def fit_k(
    pairs: str = typer.Option(..., "--pairs", "-p", help="json with {k:[...], loss:[...]}"),
    out: str = typer.Option(..., "--out", "-o"),
):
    """Fit L(k) = L∞ + A/(k+b)."""
    from ..scaling_law import fit_k_law
    obj = json.loads(Path(pairs).read_text())
    res = fit_k_law(obj["k"], obj["loss"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    t = Table(title="scaling-law fit", border_style="green")
    t.add_column("param", style="cyan")
    t.add_column("value", justify="right")
    for k, v in res.items():
        t.add_row(k, f"{v:.4f}")
    console.print(t)


@fit_app.command("size")
def fit_size(
    per_size: str = typer.Option(..., "--per-size", "-p"),
    out: str = typer.Option(..., "--out", "-o"),
):
    """Fit L∞(N) = L* + B·N^-β and A(N) = A0·N^-γ."""
    from ..scaling_law import fit_size_law
    obj = json.loads(Path(per_size).read_text())
    res = fit_size_law(obj)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2))
    console.print(json.dumps(res, indent=2))


# ──────────────────────────── report / analyze / status ────────────────────────────

@app.command("report")
def report(results_dir: str = typer.Option("results", "--results-dir", "-r")):
    """Build results/REPORT.md from result JSONs."""
    from ..report import build_report
    p = build_report(results_dir)
    console.print(f"[green]wrote[/] {p}")


@app.command("analyze")
def analyze(paths: list[str] = typer.Argument(..., help="result JSON paths")):
    """Categorize MATH-500 failures per result file."""
    from ..eval.analyze import analyze as _analyze
    for p in paths:
        res = _analyze(p)
        t = Table(title=p, border_style="green")
        t.add_column("category", style="cyan")
        t.add_column("count", justify="right")
        t.add_column("pct", justify="right")
        for cat, n in res["categories"].items():
            t.add_row(cat, str(n), res["pct"].get(cat, "—"))
        console.print(t)


@app.command("models")
def models():
    """List model presets usable via `--model <alias>`."""
    t = Table(title="model presets", border_style="cyan")
    t.add_column("alias", style="cyan")
    t.add_column("HF id")
    t.add_column("hidden", justify="right")
    t.add_column("layers", justify="right")
    t.add_column("family", style="dim")
    for alias, meta in MODEL_PRESETS.items():
        t.add_row(alias, meta["hf"], str(meta["hidden"]), str(meta["layers"]), meta["family"])
    console.print(t)
    console.print(
        "[dim]usage:[/]  [cyan]lb train sft -M qwen3-4b[/]  •  "
        "[cyan]lb eval gsm8k -M qwen3-0.6b -a runs/.../final[/]\n"
        "[dim]any HF id also accepted: -M Qwen/Qwen3-32B[/]"
    )


@app.command("status")
def status():
    """Snapshot of runs/, results/, GPU."""
    import subprocess

    cwd = Path.cwd()
    runs = sorted((cwd / "runs").glob("*")) if (cwd / "runs").exists() else []
    results = sorted((cwd / "results").glob("*.json")) if (cwd / "results").exists() else []

    t = Table(title="runs/", border_style="cyan")
    t.add_column("name")
    t.add_column("mtime", style="dim")
    for r in runs[-10:]:
        import datetime
        m = datetime.datetime.fromtimestamp(r.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        t.add_row(r.name, m)
    console.print(t)

    t2 = Table(title="results/", border_style="cyan")
    t2.add_column("file")
    t2.add_column("acc", justify="right")
    t2.add_column("n", justify="right")
    for f in results:
        try:
            d = json.loads(f.read_text())
            t2.add_row(f.name, f"{d.get('accuracy', 0):.3f}", str(d.get("n", "—")))
        except Exception:
            t2.add_row(f.name, "—", "—")
    console.print(t2)

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            console.print(Panel(out.stdout.strip(), title="GPU", border_style="yellow"))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass


if __name__ == "__main__":
    app(prog_name="lb")
