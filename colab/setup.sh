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

# ── 3. core deps ───────────────────────────────────────────────────────────────
# On Colab: install into the system Python (already has torch+CUDA).
# Elsewhere: create a project venv via `uv sync`.
IS_COLAB=0
if [[ -d /content ]] || python -c "import google.colab" 2>/dev/null; then
    IS_COLAB=1
fi
echo "[setup] colab=$IS_COLAB"

if [[ "$IS_COLAB" == "1" ]]; then
    echo "[setup] installing core deps into Colab system Python (preserves torch+CUDA)"
    uv pip install --system --no-deps -r <(python - <<'PY'
import tomllib, pathlib
p = pathlib.Path("pyproject.toml")
data = tomllib.loads(p.read_text())
for d in data["project"]["dependencies"]:
    if d.split(">=")[0].split("==")[0].split("<")[0].strip() in {"torch"}:
        continue            # keep Colab pre-installed torch+CUDA
    print(d)
PY
)
    uv pip install --system peft trl datasets accelerate transformers safetensors
else
    echo "[setup] installing project deps via uv sync"
    uv sync --no-dev
fi

# ── 4. unsloth (optional, requires CUDA) ───────────────────────────────────────
if [[ "$WITH_UNSLOTH" == "1" ]]; then
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "[setup] installing unsloth (CUDA detected)"
        # Use Colab system Python install — avoids venv torch swap.
        if [[ "$IS_COLAB" == "1" ]]; then
            uv pip install --system "unsloth" "unsloth_zoo"
        else
            uv pip install "unsloth" "unsloth_zoo"
        fi
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
if [[ "$IS_COLAB" == "1" ]]; then
    python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')"
    python -c "from src.cli.app import app; print('lb CLI: OK')"
else
    uv run python -c "import torch; print(f'torch={torch.__version__}  cuda={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}')"
    uv run python -c "from src.cli.app import app; print('lb CLI: OK')"
fi

echo
echo "✓ setup done.  cd $REPO_DIR/lora-bench  &&  ./lb --help"
