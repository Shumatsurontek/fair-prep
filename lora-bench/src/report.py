"""Generate results/REPORT.md from GSM8K + MATH-500 result JSONs."""
from __future__ import annotations

import json
from pathlib import Path

from .eval.analyze import categorize


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(d) -> str:
    if not d:
        return "—"
    n = d.get("n", 0)
    acc = d.get("accuracy", 0)
    lo, hi = wilson_ci(int(round(acc * n)), n)
    return f"{acc:.3f} [{lo:.3f}, {hi:.3f}]"


def _load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _fmt(v, fmt_str="{:.4f}"):
    if v is None:
        return "—"
    try:
        return fmt_str.format(v)
    except Exception:
        return str(v)


def build_report(results_dir: Path | str = "results") -> Path:
    base = Path(results_dir)
    rows = [(tag, _load(base / fn)) for tag, fn in [
        ("Baseline", "baseline.json"),
        ("SFT-LoRA", "sft.json"),
        ("SFT+DPO", "dpo.json"),
        ("SFT+GTW", "op_dpo_gtw.json"),
    ]]

    lines = []
    lines.append("# LoRA Bench — Report\n")
    lines.append("Modèle base : `Qwen/Qwen3-0.6B`\n\n")
    lines.append("## Résultats GSM8K\n")
    lines.append("| Étape | Accuracy [95% Wilson CI] | Tokens/s | Peak Mem (MB) | Wall (s) | N samples |")
    lines.append("|-------|--------------------------|----------|---------------|----------|-----------|")
    for tag, d in rows:
        if d is None:
            lines.append(f"| {tag} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {tag} | {fmt_ci(d)} | {_fmt(d.get('tokens_per_sec'), '{:.1f}')} | "
            f"{_fmt(d.get('peak_memory_mb'), '{:.0f}')} | "
            f"{_fmt(d.get('wall_seconds'), '{:.1f}')} | {d.get('n', '—')} |"
        )

    lines.append("\n## Gain absolu vs baseline\n")
    base_d = rows[0][1]
    for tag, d in rows[1:]:
        if base_d and d:
            gain = d["accuracy"] - base_d["accuracy"]
            lines.append(f"- {tag} : **{gain:+.4f}** ({gain*100:+.1f} pts)")

    lines.append("\n## Exemples de sorties\n")
    for tag, d in rows:
        if d is None:
            continue
        lines.append(f"### {tag}\n")
        for i, s in enumerate(d.get("samples", [])[:3]):
            lines.append(f"**Sample {i+1}** (gold={s['gold']}, pred={s['pred']}, ok={s['ok']})\n")
            lines.append(f"> {s['question']}\n")
            lines.append("```")
            lines.append(s["output"])
            lines.append("```\n")

    math_rows = [(tag, _load(base / fn)) for tag, fn in [
        ("Baseline", "math500_baseline.json"),
        ("SFT-LoRA", "math500_sft.json"),
        ("SFT+DPO", "math500_dpo.json"),
        ("SFT+GTW", "math500_gtw.json"),
    ]]

    if any(d for _, d in math_rows):
        lines.append("\n## MATH-500 (OOD bench)\n")
        lines.append("Dataset: `HuggingFaceH4/MATH-500`.\n")
        lines.append("| Étape | Accuracy [95% Wilson CI] | Tokens/s | Peak Mem (MB) | Wall (s) | N samples |")
        lines.append("|-------|--------------------------|----------|---------------|----------|-----------|")
        for tag, d in math_rows:
            if d is None:
                lines.append(f"| {tag} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {tag} | {fmt_ci(d)} | {_fmt(d.get('tokens_per_sec'), '{:.1f}')} | "
                f"{_fmt(d.get('peak_memory_mb'), '{:.0f}')} | "
                f"{_fmt(d.get('wall_seconds'), '{:.1f}')} | {d.get('n', '—')} |"
            )

        lines.append("\n### Catégorisation erreurs MATH-500\n")
        lines.append("| Étape | Correct | Truncation | Format | Symbolic mismatch | Reasoning |")
        lines.append("|-------|---------|------------|--------|-------------------|-----------|")
        for tag, d in math_rows:
            if d is None:
                lines.append(f"| {tag} | — | — | — | — | — |")
                continue
            cats = {"correct": 0, "truncation": 0, "format": 0,
                    "symbolic_mismatch": 0, "reasoning_numeric": 0, "reasoning_symbolic": 0}
            for s in d.get("samples", []):
                cats[categorize(s)] = cats.get(categorize(s), 0) + 1
            n = max(1, sum(cats.values()))

            def r(k):
                return f"{cats[k]} ({100*cats[k]/n:.0f}%)"

            reasoning = cats["reasoning_numeric"] + cats["reasoning_symbolic"]
            lines.append(
                f"| {tag} | {r('correct')} | {r('truncation')} | {r('format')} | "
                f"{r('symbolic_mismatch')} | {reasoning} ({100*reasoning/n:.0f}%) |"
            )

    out = base / "REPORT.md"
    out.write_text("\n".join(lines))
    print(f"[saved] {out}")
    return out
