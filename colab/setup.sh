#!/usr/bin/env bash
# Colab / cloud GPU bootstrap for fair-prep + lora-bench.
# Usage from a Colab cell:
#   !bash <(curl -sL https://raw.githubusercontent.com/Shumatsurontek/fair-prep/main/colab/setup.sh)
# Or after `git clone` already done:
#   !bash colab/setup.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Shumatsurontek/fair-prep.git}"
REPO_DIR="${REPO_DIR:-/content/fair-prep}"
BRANCH="${BRANCH:-main}"
WITH_UNSLOTH="${WITH_UNSLOTH:-1}"

# ── 1. clone or pull repo ──────────────────────────────────────────────────────
if [[ ! -d "$REPO_DIR/.git" ]]; then
    echo "[setup] cloning $REPO_URL → $REPO_DIR"
    git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$REPO_DIR"
else
    echo "[setup] repo exists, pulling latest"
    git -C "$REPO_DIR" fetch --depth 1 origin "$BRANCH"
    git -C "$REPO_DIR" reset --hard "origin/$BRANCH"
fi
cd "$REPO_DIR"

# ── 2. uv (fast pip alternative) ───────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "[setup] uv = $(uv --version)"

# ── 3. core deps via uv ────────────────────────────────────────────────────────
echo "[setup] installing project deps"
uv sync --no-dev

# ── 4. unsloth (optional, requires CUDA) ───────────────────────────────────────
if [[ "$WITH_UNSLOTH" == "1" ]]; then
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[setup] installing unsloth (CUDA detected)"
        # Colab-pinned wheel: avoids torch reinstall conflicts.
        uv pip install --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        uv pip install "unsloth_zoo>=2025.1.0" "xformers<0.0.27" "peft" "trl" "bitsandbytes"
    else
        echo "[setup] no CUDA → skip unsloth"
    fi
fi

# ── 5. HF auth (optional — needed for gated models) ───────────────────────────
if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "[setup] HF_TOKEN set, logging in"
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

# ── 6. expose `lb` CLI in PATH ─────────────────────────────────────────────────
chmod +x "$REPO_DIR/lora-bench/lb"
ln -sf "$REPO_DIR/lora-bench/lb" "$HOME/.local/bin/lb" 2>/dev/null || true
echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"

# ── 7. smoke test ──────────────────────────────────────────────────────────────
cd "$REPO_DIR/lora-bench"
echo "[setup] smoke check"
uv run python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')"
uv run python -c "from src.cli.app import app; print('lb CLI: OK')"

echo
echo "✓ setup done.  cd $REPO_DIR/lora-bench  &&  ./lb --help"
