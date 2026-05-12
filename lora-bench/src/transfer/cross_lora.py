"""Project teacher LoRA adapter into student hidden space.

For a target Linear module with teacher LoRA (A_t [r, d_t_in], B_t [d_t_out, r]):
    1. Collect input activations X_t (N, d_t_in), X_s (N, d_s_in) on calib set.
    2. Collect output activations Y_t (N, d_t_out), Y_s (N, d_s_out).
    3. Solve OLS for Q_in (d_s_in, d_t_in) s.t. X_s ≈ X_t · Q_in^T.
    4. Solve OLS for Q_out (d_s_out, d_t_out) s.t. Y_s ≈ Y_t · Q_out^T.
    5. Project: A_s = A_t @ pinv(Q_in^T), B_s = Q_out @ B_t.

Outputs a PEFT-style adapter dir consumable by PeftModel.from_pretrained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..core.hooks import collect_activations


def _ols_proj(X_src: torch.Tensor, X_tgt: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
    """Solve P (d_src, d_tgt) s.t. X_src @ P ≈ X_tgt. Ridge normal eq."""
    d = X_src.shape[1]
    XtX = X_src.T @ X_src
    XtX += ridge * torch.eye(d, dtype=XtX.dtype, device=XtX.device)
    return torch.linalg.solve(XtX, X_src.T @ X_tgt)


def project_lora_module(
    A_t: torch.Tensor, B_t: torch.Tensor,
    X_t: torch.Tensor, X_s: torch.Tensor,
    Y_t: torch.Tensor, Y_s: torch.Tensor,
    ridge: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Project (A_t, B_t) to student dims. Returns (A_s, B_s, recon_err)."""
    Q_in_T = _ols_proj(X_t, X_s, ridge)          # (d_t_in, d_s_in)
    Q_out_T = _ols_proj(Y_t, Y_s, ridge)         # (d_t_out, d_s_out)

    Q_in_pinv = torch.linalg.pinv(Q_in_T.T)      # (d_t_in, d_s_in)
    A_s = A_t @ Q_in_pinv                        # (r, d_s_in)
    B_s = Q_out_T.T @ B_t                        # (d_s_out, r)

    delta_t = X_t @ A_t.T @ B_t.T
    delta_s = X_s @ A_s.T @ B_s.T
    delta_t_proj = delta_t @ Q_out_T
    err = (delta_s - delta_t_proj).norm() / (delta_t_proj.norm() + 1e-8)
    return A_s, B_s, float(err)


def load_calib_prompts(path: str | Path, n: int = 256) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            out.append(obj.get("text") or obj.get("prompt") or obj.get("question", ""))
            if len(out) >= n:
                break
    return out


_LAYER_RE = __import__("re").compile(r"\.layers\.(\d+)\.")


def _peft_target_names(model, target_modules: list[str]) -> list[str]:
    names = []
    for n, _ in model.named_modules():
        if n.split(".")[-1] in target_modules:
            names.append(n)
    return names


def _layer_idx(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _module_leaf(name: str) -> str:
    return name.split(".")[-1]


def _build_layer_map(n_student: int, n_teacher: int) -> dict[int, int]:
    """Linear stride: student layer j → teacher layer round(j · (L_t-1)/(L_s-1))."""
    if n_student <= 1:
        return {0: 0}
    return {j: round(j * (n_teacher - 1) / (n_student - 1)) for j in range(n_student)}


def transfer_adapter(
    teacher_name: str,
    student_name: str,
    adapter_path: str,
    calib_path: str,
    out_dir: str,
    device: str = "cuda",
    n_calib: int = 256,
    ridge: float = 1e-4,
    max_length: int = 256,
) -> dict:
    """End-to-end transfer. Returns dict of per-module recon errors."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    lora_cfg = LoraConfig.from_pretrained(adapter_path)
    target_modules = list(lora_cfg.target_modules)

    tok_t = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(student_name, trust_remote_code=True)
    for tok in (tok_t, tok_s):
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    prompts = load_calib_prompts(calib_path, n_calib)

    # Read layer counts from config (JSON only — no model weights loaded).
    L_s = AutoConfig.from_pretrained(student_name, trust_remote_code=True).num_hidden_layers
    L_t = AutoConfig.from_pretrained(teacher_name, trust_remote_code=True).num_hidden_layers
    layer_map = _build_layer_map(L_s, L_t)

    # T4 (Turing) lacks bf16 Tensor Cores → use fp16 (faster).
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.is_bf16_supported()
             else torch.float16 if device == "cuda" else torch.float32)
    print(f"[transfer] dtype={dtype}")

    # Now load student model.
    student = AutoModelForCausalLM.from_pretrained(
        student_name, trust_remote_code=True, dtype=dtype
    ).to(device)
    teacher_layers_used = set(layer_map.values())
    print(f"[layer-map] student L={L_s} → teacher L={L_t}  ({len(teacher_layers_used)} unique teacher layers)")

    # Collect student activations first (smaller model, less memory pressure).
    target_names_s_all = _peft_target_names(student, target_modules)
    target_names_s = [n for n in target_names_s_all
                      if _layer_idx(n) is not None and _layer_idx(n) < L_s]
    acts_s_raw = collect_activations(student, tok_s, prompts, target_names_s, device,
                                     max_length=max_length)
    del student
    if device == "cuda":
        torch.cuda.empty_cache()

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_name, trust_remote_code=True, dtype=dtype
    ).to(device)
    teacher_peft = PeftModel.from_pretrained(teacher, adapter_path)
    # Only hook teacher layers we actually need.
    target_names_t_all = _peft_target_names(teacher_peft, target_modules)
    target_names_t = [n for n in target_names_t_all if _layer_idx(n) in teacher_layers_used]
    print(f"[hooks] teacher={len(target_names_t)}  student={len(target_names_s)}")
    acts_t_raw = collect_activations(teacher_peft, tok_t, prompts, target_names_t, device,
                                     max_length=max_length)
    lora_weights = {}
    for n, m in teacher_peft.named_modules():
        if hasattr(m, "lora_A") and _module_leaf(n) in target_modules:
            A = m.lora_A["default"].weight.detach().to("cpu", torch.float32)
            B = m.lora_B["default"].weight.detach().to("cpu", torch.float32)
            base_name = n.replace("base_model.model.", "")
            lora_weights[base_name] = (A, B)
    del teacher_peft, teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    def _strip(n: str) -> str:
        return n.replace("base_model.model.", "")
    acts_t = {_strip(k): v for k, v in acts_t_raw.items()}
    acts_s = {_strip(k): v for k, v in acts_s_raw.items()}

    # Pair student modules to teacher modules via layer_map + leaf name.
    # Student name = "model.layers.{j}.<...>.<leaf>".  Map to teacher_i = layer_map[j].
    from tqdm import tqdm
    errs: dict[str, float] = {}
    out_state: dict[str, torch.Tensor] = {}
    print(f"[ols-solve] projecting {len(acts_s)} student modules…")
    for s_name in tqdm(list(acts_s), desc="ols solve"):
        j = _layer_idx(s_name)
        leaf = _module_leaf(s_name)
        if j is None or j not in layer_map:
            continue
        i = layer_map[j]
        # Find teacher key matching layer i + same leaf
        t_name = next((tn for tn in acts_t
                       if _layer_idx(tn) == i and _module_leaf(tn) == leaf), None)
        if t_name is None or t_name not in lora_weights:
            continue
        A_t, B_t = lora_weights[t_name]
        X_t, Y_t = acts_t[t_name]
        X_s, Y_s = acts_s[s_name]
        n = min(X_t.shape[0], X_s.shape[0])
        # Upcast to fp32 for OLS solve.
        A_s, B_s, err = project_lora_module(
            A_t.float(), B_t.float(),
            X_t[:n].float(), X_s[:n].float(),
            Y_t[:n].float(), Y_s[:n].float(),
            ridge=ridge,
        )
        errs[s_name] = err
        out_state[f"base_model.model.{s_name}.lora_A.weight"] = A_s.contiguous()
        out_state[f"base_model.model.{s_name}.lora_B.weight"] = B_s.contiguous()

    save_file(out_state, str(out_p / "adapter_model.safetensors"))
    LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        lora_dropout=lora_cfg.lora_dropout,
        target_modules=target_modules,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    ).save_pretrained(str(out_p))

    with open(out_p / "transfer_errors.json", "w") as f:
        json.dump({"mean_err": sum(errs.values()) / max(len(errs), 1),
                   "per_module": errs}, f, indent=2)
    return errs
