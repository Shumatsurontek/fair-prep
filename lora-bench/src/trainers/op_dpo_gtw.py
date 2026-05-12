"""OnPolicy-DPO-GTW: Group-Token-Weighted Direct Preference Optimization.

Novel method: combine online generation + verifier-based pairing + token-level
credit derived via Group Leave-One-Out (LOO) attribution. Free process
supervision without training a PRM.

Pipeline per step
-----------------
1. Sample G completions per prompt from current policy π_θ.
2. Score with reward_fn (e.g. GSM8K exact-match).
3. Filter prompts with zero reward variance (no signal).
4. Build (chosen=correct, rejected=incorrect) pairs intra-group.
5. Compute per-token weight w_t = position-wise inter-group variance · r_chosen.
   Tokens shared across group members carry no signal (w≈0). Tokens that
   diverge between correct and incorrect carry high w.
6. Loss = α₁·DPO_sigmoid + α₂·DPO_token_weighted + α₃·SFT_chosen + α₄·KL.
7. β adaptatif cosine schedule + optional replay buffer injection.

Status: research prototype. Smoke-test on MPS with `MAX_STEPS=10 NUM_GEN=2`.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from trl import DPOTrainer

from .replay_buffer import Pair, ReplayBuffer


@dataclass
class OPDPOGTWConfig:
    """Knobs specific to OnPolicy-DPO-GTW. Extends DPOConfig usage."""
    num_generations: int = 4
    gen_max_new_tokens: int = 512
    gen_temperature: float = 0.8
    gen_top_p: float = 0.9

    # Loss combination
    w_dpo_sigmoid: float = 1.0
    w_token_dpo: float = 0.5
    w_sft: float = 0.3
    w_kl: float = 0.0  # explicit KL on top of DPO implicit one; 0 by default

    # Beta schedule
    beta_schedule: str = "cosine"  # "cosine" | "constant"
    beta_0: float = 0.2
    beta_min: float = 0.02

    # Replay buffer
    replay_enabled: bool = True
    replay_size: int = 256
    replay_inject_prob: float = 0.1

    # Token weighting
    token_weight_floor: float = 0.05  # avoid w=0 killing all signal


def _decode_completions(tokenizer, full_ids: torch.Tensor, prompt_lens: list[int]) -> list[str]:
    out = []
    for i, plen in enumerate(prompt_lens):
        seq = full_ids[i][plen:]
        out.append(tokenizer.decode(seq, skip_special_tokens=True))
    return out


def compute_group_loo_token_weights(
    chosen_ids: torch.Tensor,  # (B, T)
    group_ids_list: list[torch.Tensor],  # list len G of (B, T)
    pad_id: int,
    floor: float = 0.05,
) -> torch.Tensor:
    """Per-token weight = variance over group of `token == chosen_token` indicator.

    Shape: (B, T). Tokens shared by all group members have variance 0 → low weight
    (set to floor). Tokens where chosen differs from the rest carry high weight.

    Assumes all group_ids_list tensors have been right-padded to same T as chosen.
    """
    B, T = chosen_ids.shape
    chosen_expanded = chosen_ids.unsqueeze(0)  # (1, B, T)
    stack = torch.stack(group_ids_list, dim=0)  # (G, B, T)
    match = (stack == chosen_expanded).float()  # (G, B, T)
    var = match.var(dim=0, unbiased=False)  # (B, T)
    pad_mask = (chosen_ids != pad_id).float()
    w = (var * pad_mask).clamp(min=floor)
    # Normalize per-sequence so weights sum to length of valid tokens (preserves scale)
    seq_len = pad_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    w = w * seq_len / w.sum(dim=1, keepdim=True).clamp(min=1e-6)
    return w * pad_mask


def cosine_beta(step: int, max_steps: int, beta_0: float, beta_min: float) -> float:
    if max_steps <= 0:
        return beta_0
    progress = min(1.0, max(0.0, step / max_steps))
    return beta_min + (beta_0 - beta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))


class OnPolicyDPOGTWTrainer(DPOTrainer):
    """Custom DPOTrainer with online generation, verifier scoring, token-LOO weights."""

    def __init__(
        self,
        *args,
        op_cfg: OPDPOGTWConfig,
        reward_fn: Callable[..., list[float]],
        prompt_dataset: Dataset,  # prompts + ground_truth columns
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.op_cfg = op_cfg
        self.reward_fn = reward_fn
        self.prompt_dataset = prompt_dataset
        self.replay = (
            ReplayBuffer(size=op_cfg.replay_size, seed=self.args.seed)
            if op_cfg.replay_enabled else None
        )
        self._iter_idx = 0

    # ---- adaptive beta ----
    def current_beta(self) -> float:
        if self.op_cfg.beta_schedule == "cosine":
            return cosine_beta(
                step=self.state.global_step,
                max_steps=self.state.max_steps or 1,
                beta_0=self.op_cfg.beta_0,
                beta_min=self.op_cfg.beta_min,
            )
        return self.op_cfg.beta_0

    # ---- online generation step ----
    @torch.no_grad()
    def _generate_group(self, prompts_text: list[str]) -> list[list[str]]:
        tok = self.processing_class
        model = self.model
        model.eval()
        device = next(model.parameters()).device
        G = self.op_cfg.num_generations
        outputs = []
        for prompt in prompts_text:
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=self.op_cfg.gen_max_new_tokens,
                do_sample=True,
                temperature=self.op_cfg.gen_temperature,
                top_p=self.op_cfg.gen_top_p,
                num_return_sequences=G,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            plen = int(enc["input_ids"].shape[1])
            comps = [tok.decode(gen[i, plen:], skip_special_tokens=True) for i in range(G)]
            outputs.append(comps)
        model.train()
        return outputs

    def _build_pairs(
        self,
        prompts: list[str],
        groups: list[list[str]],
        ground_truths: list[str],
    ) -> list[Pair]:
        pairs: list[Pair] = []
        for prompt, gens, gt in zip(prompts, groups, ground_truths):
            rewards = self.reward_fn(
                prompts=[prompt] * len(gens),
                completions=gens,
                ground_truth=[gt] * len(gens),
            )
            corrects = [g for g, r in zip(gens, rewards) if r > 0.5]
            incorrects = [g for g, r in zip(gens, rewards) if r <= 0.5]
            if not corrects or not incorrects:
                continue
            for c in corrects:
                for w in incorrects:
                    pairs.append(Pair(
                        prompt=prompt, chosen=c, rejected=w,
                        reward_margin=1.0, step_added=self.state.global_step,
                    ))
        return pairs

    # ---- one online step builds a synthetic preference dataset and forwards ----
    def online_step(self, batch_prompts: list[dict]) -> dict:
        prompts_text = [b["prompt"] for b in batch_prompts]
        gts = [b["ground_truth"] for b in batch_prompts]
        groups = self._generate_group(prompts_text)
        pairs = self._build_pairs(prompts_text, groups, gts)
        if self.replay is not None:
            replay_pairs = self.replay.maybe_inject(
                batch_size=len(pairs) or 1,
                inject_prob=self.op_cfg.replay_inject_prob,
            )
            pairs.extend(replay_pairs)
            self.replay.push_many([
                Pair(prompt=p.prompt, chosen=p.chosen, rejected=p.rejected,
                     reward_margin=p.reward_margin, step_added=p.step_added)
                for p in pairs[-(len(pairs) - len(replay_pairs)):]
            ])
        return {
            "pairs": pairs,
            "n_correct": sum(1 for g, gs in zip(groups, groups) for c in gs),
            "frac_zero_std": 1.0 - (len(pairs) > 0),
        }


# Notes on integration with HuggingFace Trainer loop
# --------------------------------------------------
# The cleanest way to plug online generation into a Trainer subclass is to
# override `get_train_dataloader()` to return an IterableDataset that yields
# freshly generated pairs each iteration. To keep this prototype concise we
# also provide a fully self-contained `StandaloneOnPolicyDPOGTW` below that
# manages its own optimization loop (no Trainer.train()).


# =============================================================================
# Standalone runnable trainer — does NOT inherit DPOTrainer.
# Manages its own optimizer / scheduler / logging / checkpointing.
# =============================================================================


def _logp_completion(
    model, tokenizer, prompt_text: str, completion_text: str,
    device: str, with_adapter: bool = True,
) -> torch.Tensor:
    """Per-token log p(completion_t | prompt, completion_<t) under model."""
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    comp_ids = tokenizer(completion_text, return_tensors="pt",
                         add_special_tokens=False).input_ids.to(device)
    if comp_ids.shape[1] == 0:
        return torch.zeros(0, device=device)
    full = torch.cat([prompt_ids, comp_ids], dim=1)
    T_prompt = prompt_ids.shape[1]

    ctx = model.disable_adapter() if not with_adapter else _noop_ctx()
    with ctx:
        out = model(full)
    logits = out.logits                                       # [1, T_full, V]
    targets = full[:, 1:]                                     # [1, T_full-1]
    logp_all = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    logp_tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
    return logp_tok[T_prompt - 1: full.shape[1] - 1]          # [T_completion]


class _noop_ctx:
    def __enter__(self): return self
    def __exit__(self, *exc): return False


class StandaloneOnPolicyDPOGTW:
    """Fully self-managed loop. Uses helpers above (cosine_beta, token weights)."""

    def __init__(
        self,
        model,                       # PEFT LoRA model
        tokenizer,
        prompt_dataset,              # iterable of {"prompt": str, "ground_truth": str}
        reward_fn,                   # (prompts, completions, gold) -> list[float]
        op_cfg: OPDPOGTWConfig,
        train_cfg: dict,             # {output_dir, max_steps, lr, warmup_steps, ...}
        device: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompts = list(prompt_dataset)
        self.reward_fn = reward_fn
        self.op_cfg = op_cfg
        self.cfg = train_cfg
        self.device = device

        self.max_steps = int(train_cfg["max_steps"])
        self.optim = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.sched = CosineAnnealingLR(
            self.optim,
            T_max=max(1, self.max_steps - int(train_cfg.get("warmup_steps", 0))),
        )
        self.replay = ReplayBuffer(size=op_cfg.replay_size, seed=int(train_cfg.get("seed", 42))) \
            if op_cfg.replay_enabled else None
        self.run_dir = Path(train_cfg["output_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(self.run_dir / "tb")
        except Exception:
            self.tb = None

        self.global_step = 0
        self.history: list[dict] = []

    # ---- main loop ----
    def train(self):
        loop = itertools.cycle(self.prompts)
        prompt_bs = int(self.cfg.get("prompt_batch_size", 2))
        for step in range(self.max_steps):
            t0 = time.perf_counter()
            self.global_step = step
            batch = [next(loop) for _ in range(prompt_bs)]
            metrics = self._step(batch)
            metrics["wall_s"] = time.perf_counter() - t0
            metrics["lr"] = self.optim.param_groups[0]["lr"]
            self._log(metrics)
            if (step + 1) % int(self.cfg.get("save_every", 100)) == 0:
                self._save(step + 1)
        self._save(self.max_steps, final=True)

    # ---- one optimization step ----
    def _step(self, batch_prompts):
        beta = cosine_beta(self.global_step, self.max_steps, self.op_cfg.beta_0, self.op_cfg.beta_min) \
            if self.op_cfg.beta_schedule == "cosine" else self.op_cfg.beta_0

        all_pairs: list[Pair] = []
        all_chosen_w: list[torch.Tensor] = []
        all_rewards: list[float] = []
        for ex in batch_prompts:
            comps = self._sample_group(ex["prompt"])
            rewards = self.reward_fn(
                prompts=[ex["prompt"]] * len(comps),
                completions=comps,
                ground_truth=[ex["ground_truth"]] * len(comps),
            )
            all_rewards.extend(rewards)
            if sum(rewards) == 0 or sum(rewards) == len(rewards):
                continue
            pairs, weights = self._pairs_with_weights(ex["prompt"], comps, rewards)
            all_pairs.extend(pairs)
            all_chosen_w.extend(weights)

        # replay
        if self.replay is not None and len(self.replay) > 0:
            extra = self.replay.maybe_inject(max(1, len(all_pairs)), self.op_cfg.replay_inject_prob)
            for p in extra:
                all_pairs.append(p)
                all_chosen_w.append(torch.zeros(0))

        metrics = {
            "step": self.global_step,
            "beta": beta,
            "n_pairs": len(all_pairs),
            "mean_reward": float(sum(all_rewards) / max(1, len(all_rewards))),
            "loss": 0.0, "loss_dpo_seq": 0.0, "loss_dpo_tok": 0.0,
            "loss_sft": 0.0, "loss_kl": 0.0,
        }
        if not all_pairs:
            return metrics

        losses = self._losses(all_pairs, all_chosen_w, beta)
        total = (
            self.op_cfg.w_dpo_sigmoid * losses["dpo_seq"]
            + self.op_cfg.w_token_dpo * losses["dpo_tok"]
            + self.op_cfg.w_sft * losses["sft"]
            + self.op_cfg.w_kl * losses["kl"]
        )

        self.optim.zero_grad(set_to_none=True)
        total.backward()
        clip = float(self.cfg.get("max_grad_norm", 1.0))
        if clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
        self.optim.step()
        if self.global_step >= int(self.cfg.get("warmup_steps", 0)):
            self.sched.step()

        if self.replay is not None:
            for p in all_pairs[: min(8, len(all_pairs))]:
                self.replay.push(p)

        metrics["loss"] = float(total.detach())
        metrics["loss_dpo_seq"] = float(losses["dpo_seq"].detach())
        metrics["loss_dpo_tok"] = float(losses["dpo_tok"].detach())
        metrics["loss_sft"] = float(losses["sft"].detach())
        metrics["loss_kl"] = float(losses["kl"].detach())
        return metrics

    @torch.no_grad()
    def _sample_group(self, raw_question: str) -> list[str]:
        from ..data import format_qwen_prompt_only
        self.model.eval()
        rendered = format_qwen_prompt_only(raw_question, self.tokenizer)
        enc = self.tokenizer(rendered, return_tensors="pt").to(self.device)
        outs = []
        for _ in range(self.op_cfg.num_generations):
            gen = self.model.generate(
                **enc,
                max_new_tokens=self.op_cfg.gen_max_new_tokens,
                do_sample=True,
                temperature=self.op_cfg.gen_temperature,
                top_p=self.op_cfg.gen_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            outs.append(self.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:],
                                              skip_special_tokens=True))
        self.model.train()
        return outs

    def _pairs_with_weights(self, raw_q, completions, rewards):
        enc = [self.tokenizer(c, return_tensors="pt", add_special_tokens=False).input_ids[0]
               for c in completions]
        lengths = [e.shape[0] for e in enc]
        T_max = max(lengths) if lengths else 0
        if T_max == 0:
            return [], []
        padded = torch.full((len(enc), T_max), int(self.tokenizer.pad_token_id), dtype=torch.long)
        for i, e in enumerate(enc):
            padded[i, : e.shape[0]] = e

        pairs, weights = [], []
        for i, j in itertools.product(range(len(completions)), repeat=2):
            if rewards[i] <= rewards[j]:
                continue
            pairs.append(Pair(
                prompt=raw_q, chosen=completions[i], rejected=completions[j],
                reward_margin=float(rewards[i] - rewards[j]),
                step_added=self.global_step,
            ))
            # token weights of the chosen
            w = compute_group_loo_token_weights(
                chosen_ids=padded[i].unsqueeze(0),
                group_ids_list=[padded[k].unsqueeze(0) for k in range(len(completions))],
                pad_id=int(self.tokenizer.pad_token_id),
                floor=self.op_cfg.token_weight_floor,
            ).squeeze(0)[: lengths[i]] * float(rewards[i])
            weights.append(w.detach())
        return pairs, weights

    def _losses(self, pairs, chosen_weights, beta):
        L_dpo_seq = torch.tensor(0.0, device=self.device)
        L_dpo_tok = torch.tensor(0.0, device=self.device)
        L_sft = torch.tensor(0.0, device=self.device)
        L_kl = torch.tensor(0.0, device=self.device)
        n = 0
        for pair, w in zip(pairs, chosen_weights):
            try:
                lp_w_pol = _logp_completion(self.model, self.tokenizer, pair.prompt, pair.chosen,
                                            self.device, with_adapter=True)
                lp_l_pol = _logp_completion(self.model, self.tokenizer, pair.prompt, pair.rejected,
                                            self.device, with_adapter=True)
                with torch.no_grad():
                    lp_w_ref = _logp_completion(self.model, self.tokenizer, pair.prompt, pair.chosen,
                                                self.device, with_adapter=False)
                    lp_l_ref = _logp_completion(self.model, self.tokenizer, pair.prompt, pair.rejected,
                                                self.device, with_adapter=False)
            except Exception as e:                                # noqa: BLE001
                print(f"[warn] logp failed: {e}")
                continue
            if lp_w_pol.numel() == 0 or lp_l_pol.numel() == 0:
                continue
            r_w = lp_w_pol.sum() - lp_w_ref.sum()
            r_l = lp_l_pol.sum() - lp_l_ref.sum()
            L_dpo_seq = L_dpo_seq - F.logsigmoid(beta * (r_w - r_l))

            if w.numel() > 0:
                T = min(lp_w_pol.shape[0], w.shape[0])
                w_t = w[:T].to(self.device)
                diff_w = lp_w_pol[:T] - lp_w_ref[:T]
                T2 = min(lp_l_pol.shape[0], T)
                diff_l = lp_l_pol[:T2] - lp_l_ref[:T2]
                weighted = (w_t[:T] * diff_w).sum() - (w_t[:T2] * diff_l).sum()
                L_dpo_tok = L_dpo_tok - F.logsigmoid(beta * weighted)

            L_sft = L_sft - lp_w_pol.mean()
            L_kl = L_kl + (lp_w_pol.detach() - lp_w_ref).mean()
            n += 1

        n = max(1, n)
        return {"dpo_seq": L_dpo_seq / n, "dpo_tok": L_dpo_tok / n,
                "sft": L_sft / n, "kl": L_kl / n}

    def _log(self, m):
        self.history.append(m)
        print(f"[step {m['step']:4d}] pairs={m['n_pairs']} R̄={m['mean_reward']:.3f} "
              f"β={m['beta']:.3f} L={m['loss']:.4f} (seq={m['loss_dpo_seq']:.3f} "
              f"tok={m['loss_dpo_tok']:.3f} sft={m['loss_sft']:.3f} kl={m['loss_kl']:.3f}) "
              f"{m['wall_s']:.1f}s")
        if self.tb is not None:
            for k, v in m.items():
                if k == "step":
                    continue
                self.tb.add_scalar(k, v, m["step"])

    def _save(self, step, final=False):
        import json
        out = self.run_dir / ("final" if final else f"checkpoint-{step}")
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out))
        self.tokenizer.save_pretrained(str(out))
        (self.run_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        print(f"[save] {out}")


# =============================================================================
# Pure run() entry-point for CLI dispatcher.
# =============================================================================

def _build_prompt_dataset(name: str, split: str, n: int) -> list[dict]:
    from datasets import load_dataset
    from ..data import extract_gsm8k_answer

    raw = load_dataset(name, "main", split=split)
    if n and n < len(raw):
        raw = raw.select(range(n))
    out = []
    for ex in raw:
        _, gold = extract_gsm8k_answer(ex["answer"])
        out.append({"prompt": ex["question"], "ground_truth": gold})
    return out


def run(
    cfg: dict,
    device: str,
    sft_checkpoint: str,
    max_steps: int | None = None,
    max_train_prompts: int | None = None,
):
    from peft import PeftModel

    from ..core.model import load_base_model, load_tokenizer
    from ..core.utils import ensure_dir, log_env, save_json
    from ..rewards import get_reward_fn

    if max_steps is not None:
        cfg["train"]["max_steps"] = max_steps
    if max_train_prompts is not None:
        cfg["dataset"]["max_train_samples"] = max_train_prompts

    run_dir = ensure_dir(Path(cfg["output_root"]) / cfg["run_name"])
    save_json({"env": log_env(device), "cfg": cfg, "sft_ckpt": sft_checkpoint},
              run_dir / "run_meta.json")

    tokenizer = load_tokenizer(cfg)
    base = load_base_model(cfg, device)
    model = PeftModel.from_pretrained(base, sft_checkpoint, is_trainable=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable={n_train:,}")

    n_prompts = int(cfg["dataset"].get("max_train_samples") or 200)
    prompt_ds = _build_prompt_dataset(
        cfg["dataset"].get("name", "openai/gsm8k"),
        cfg["dataset"].get("train_split", "train"),
        n=n_prompts,
    )
    print(f"[data] n_prompts={len(prompt_ds)}")

    reward_fn_raw = get_reward_fn(cfg["reward"]["type"])

    def reward_fn(prompts, completions, ground_truth):
        return reward_fn_raw(prompts, completions, ground_truth=ground_truth)

    op_cfg = OPDPOGTWConfig(
        num_generations=int(cfg["generation"]["num_generations"]),
        gen_max_new_tokens=int(cfg["generation"]["max_new_tokens"]),
        gen_temperature=float(cfg["generation"]["temperature"]),
        gen_top_p=float(cfg["generation"]["top_p"]),
        w_dpo_sigmoid=float(cfg["loss_weights"]["dpo_sigmoid"]),
        w_token_dpo=float(cfg["loss_weights"]["token_dpo"]),
        w_sft=float(cfg["loss_weights"]["sft"]),
        w_kl=float(cfg["loss_weights"].get("kl", 0.0)),
        beta_schedule=cfg["beta"]["schedule"],
        beta_0=float(cfg["beta"]["beta_0"]),
        beta_min=float(cfg["beta"]["beta_min"]),
        replay_enabled=bool(cfg["replay_buffer"]["enabled"]),
        replay_size=int(cfg["replay_buffer"]["size"]),
        replay_inject_prob=float(cfg["replay_buffer"]["inject_prob"]),
        token_weight_floor=float(cfg.get("token_weight_floor", 0.05)),
    )

    train_cfg = {**cfg["train"], "output_dir": str(run_dir), "seed": cfg.get("seed", 42)}

    trainer = StandaloneOnPolicyDPOGTW(
        model=model, tokenizer=tokenizer,
        prompt_dataset=prompt_ds, reward_fn=reward_fn,
        op_cfg=op_cfg, train_cfg=train_cfg, device=device,
    )
    trainer.train()
    return run_dir / "final"
