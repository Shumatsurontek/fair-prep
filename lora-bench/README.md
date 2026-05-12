# lora-bench

Boilerplate pour fine-tuner **Qwen3-0.6B** par **LoRA** (SFT puis DPO) et mesurer empiriquement les gains de chaque étape sur **GSM8K**.

Pipeline : `Qwen3-0.6B (base)` → eval → `+ LoRA SFT GSM8K` → eval → `+ DPO Math-Step` → eval → **`+ OnPolicy-DPO-GTW (custom)`** → eval → REPORT.md.

---

## Quickstart

Depuis la racine `fair-prep/` :

```bash
# 1. Install deps
cd lora-bench
bash scripts/00_setup.sh                # uv sync + pre-download model & datasets

# 2. Baseline eval
bash scripts/01_eval_baseline.sh         # results/baseline.json

# 3. SFT LoRA
bash scripts/02_train_sft.sh             # ~ 30-60 min sur M-series, ~10-20 min CUDA
bash scripts/03_eval_sft.sh              # results/sft.json

# 4. DPO sur SFT checkpoint
bash scripts/04_train_dpo.sh             # ~ 20-40 min
bash scripts/05_eval_dpo.sh              # results/dpo.json

# 5. (Recherche) OnPolicy-DPO-GTW custom trainer
bash scripts/06_train_op_dpo_gtw.sh      # on-policy gen + token-LOO weights
bash scripts/07_eval_op_dpo_gtw.sh       # results/op_dpo_gtw.json

# 6. Tout d'un coup
bash scripts/99_bench_all.sh             # chaîne tout + génère REPORT.md
# Skip GTW :  SKIP_GTW=1 bash scripts/99_bench_all.sh
```

Variables d'env utiles : `DEVICE=cuda|mps|cpu`, `MAX_SAMPLES=200`, `MAX_STEPS=-1`.

Smoke test sur 5 steps SFT :

```bash
MAX_STEPS=5 MAX_SAMPLES=10 bash scripts/02_train_sft.sh
```

---

## Layout

```
lora-bench/
├── configs/
│   ├── base.yaml             # modèle, tokenizer, LoRA
│   ├── sft_gsm8k.yaml        # SFT hyperparams
│   ├── dpo_pref.yaml         # DPO hyperparams
│   └── op_dpo_gtw.yaml       # custom GTW trainer hyperparams
├── src/
│   ├── utils.py              # device, seed, config, IO
│   ├── model.py              # load_model + apply_lora
│   ├── data.py               # GSM8K + preference loaders, format Qwen chat
│   ├── train_sft.py          # TRL SFTTrainer
│   ├── train_dpo.py          # TRL DPOTrainer (+ --trainer op_dpo_gtw delegate)
│   ├── train_op_dpo_gtw.py   # launcher for custom GTW trainer
│   ├── eval.py               # GSM8K eval
│   ├── infer.py              # interactive generation
│   ├── make_report.py        # JSON -> REPORT.md (incl. GTW row)
│   ├── replay_buffer.py      # for GTW
│   ├── rewards/              # reward fns + sanity test
│   │   ├── __init__.py
│   │   └── test.py
│   └── trainers/             # custom trainers
│       ├── __init__.py
│       └── op_dpo_gtw.py     # helpers + StandaloneOnPolicyDPOGTW
├── scripts/
│   ├── 00_setup.sh
│   ├── 01_eval_baseline.sh
│   ├── 02_train_sft.sh
│   ├── 03_eval_sft.sh
│   ├── 04_train_dpo.sh
│   ├── 05_eval_dpo.sh
│   ├── 06_train_op_dpo_gtw.sh
│   ├── 07_eval_op_dpo_gtw.sh
│   └── 99_bench_all.sh
└── results/
    ├── baseline.json
    ├── sft.json
    ├── dpo.json
    ├── op_dpo_gtw.json
    └── REPORT.md
```

**GTW (research)** : voir `../maths/05_research_dpo_gtw.md` pour la formulation mathématique
(token-LOO attribution, multi-loss, β cosine, replay buffer).
Smoke test MPS : `MAX_STEPS=10 MAX_TRAIN_PROMPTS=20 bash scripts/06_train_op_dpo_gtw.sh`.

---

## Hyperparamètres clés (à connaître pour l'entretien)

### LoRA (`configs/base.yaml`)
| Param | Valeur | Pourquoi |
|-------|--------|----------|
| `r` | 16 | Rang bas-rang. Aghajanyan : intrinsic dim faible. |
| `alpha` | 32 (= 2r) | Scaling. Effective LR = lr · α/r. |
| `dropout` | 0.05 | Régularisation légère. |
| `target_modules` | qkv, o, gate/up/down | Toutes les projections d'attention + MLP. |

Trainable params : ~5M / 600M (≈ 0.8%).

### SFT (`configs/sft_gsm8k.yaml`)
| Param | Valeur | Pourquoi |
|-------|--------|----------|
| `lr` | 2e-4 | Standard LoRA. |
| `epochs` | 3 | GSM8K = 7.5k samples, 3 passes raisonnable. |
| `batch eff.` | 16 | per-device 4 × grad-accum 4. |
| `seq len` | 1024 | GSM8K reste court avec CoT. |
| `scheduler` | cosine | Decay smooth, warmup 3%. |

### DPO (`configs/dpo_pref.yaml`)
| Param | Valeur | Pourquoi |
|-------|--------|----------|
| `lr` | 5e-6 | DPO sensible, lr bas. |
| `beta` | 0.1 | Standard ; trop grand = stuck, trop petit = drift. |
| `loss_type` | sigmoid | DPO original. Alternatives : ipo, kto. |
| `epochs` | 1 | DPO tend à overfitter rapidement. |

---

## Notes hardware

### Mac (MPS)

- `bitsandbytes` non disponible → fp16 obligatoire (pas de 4-bit).
- Désactiver flash-attn (`attn_implementation=eager`) car certaines ops fallback CPU.
- DPO sur MPS : OK pour Qwen3-0.6B avec batch 2 + grad-accum 16.
- Pas de `torch.cuda.reset_peak_memory_stats` → `peak_memory_mb` peut être null sur MPS.

### CUDA (Colab T4 / A100)

- bf16 préféré sur A100/H100, sinon fp16 (T4).
- Optionnel : `optim=paged_adamw_8bit` (avec `bitsandbytes`) pour économiser mémoire optim states.
- Activer `gradient_checkpointing=True` si OOM (compromis 30% lent).

### Reproductibilité

- Seed 42 fixée partout.
- Versions pinnées via `pyproject.toml`.
- Format JSON pour les résultats, regen `REPORT.md` à tout moment.

---

## Test rapide d'inférence

```bash
uv run python -m src.infer \
    --adapter ./runs/qwen3_06b_sft_gsm8k/final \
    --question "Marie has 3 apples. She buys 5 more, then gives 2 away. How many apples does she have?"
```

---

## Métriques rapportées

- **accuracy** : exact-match du nombre extrait par regex `#### N` ou dernier nombre du texte.
- **tokens_per_sec** : tokens générés / wall time.
- **peak_memory_mb** : VRAM peak (CUDA) ou MPS allocated.
- **wall_seconds** : temps total d'éval.
- **trainable_params** : nb de params LoRA (loggé en début de training).

---

## Limites connues

- **Eval rapide** : par défaut `MAX_SAMPLES=200`. Pour le full GSM8K test (1.3k), passer `MAX_SAMPLES=1319`.
- **Pas de maj@N** : on fait greedy uniquement. Pour mieux, passer en `do_sample=True` + N tirages + voting (à ajouter dans `eval.py` si besoin).
- **DPO dataset** : `xinlai/Math-Step-DPO-10K` (math) ; basculer vers UltraFeedback si pas dispo.
- **MPS** : certaines ops Triton non supportées → on reste sur attn eager.

---

## Référence rapide pour les concepts

- LoRA = bas-rang : cours/exos `maths/03_algebre_lineaire/`
- DPO loss = régression logistique sur log-rapports : démo `maths/01_optimisation/03_corrections.md` exo 1.12
- KL/Fisher dans PPO/TRPO : exo 1.5
- Variance reduction (baseline = control variate) : exo 1.8 + 2.10
