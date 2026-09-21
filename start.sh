#!/bin/bash
set -e

echo "Starting Runpod Ollama worker..."

# Runpod's model store mounts at /runpod-volume even when no network volume is
# attached, and that mount is not documented as writable — so probe rather than
# assume. Only ever used inside `if`, so `set -e` won't kill the script.
writable() {
    mkdir -p "$1" 2>/dev/null && touch "$1/.rw-probe" 2>/dev/null && rm -f "$1/.rw-probe" 2>/dev/null
}

VOLUME_ROOT="/runpod-volume"

# From inside the worker there is no way to tell a network volume from a local
# volume disk from a model-store mount — they all appear at the same path. So
# report what is actually observable (present / writable) and say what that does
# and doesn't imply.
VOLUME_STATE="absent"
if [ -d "$VOLUME_ROOT" ]; then
    if writable "$VOLUME_ROOT/ollama"; then
        VOLUME_STATE="writable"
    else
        VOLUME_STATE="read-only"
    fi
fi

export RUNPOD_MODEL_CACHE_DIR="${RUNPOD_MODEL_CACHE_DIR:-$VOLUME_ROOT/huggingface-cache/hub}"

case "$VOLUME_STATE" in
    writable)
        echo "Volume: $VOLUME_ROOT present and writable — network volume or local volume disk (indistinguishable from here). Models cached here only survive across workers if it is a network volume."
        ;;
    read-only)
        echo "Volume: $VOLUME_ROOT present but NOT writable — likely a model-store mount only. Caches will go to container disk and will not be reused by the next worker."
        ;;
    absent)
        echo "Volume: no $VOLUME_ROOT — using container disk. Attach a network volume to reuse models across workers."
        ;;
esac

# Note this only reports that the directory exists. It may be Runpod's prefill or
# this worker's own fallback downloads, which use the same layout. The per-model
# "[ModelStore] Using snapshot" line is the authoritative signal.
if [ -d "$RUNPOD_MODEL_CACHE_DIR" ]; then
    echo "HF cache dir present: $RUNPOD_MODEL_CACHE_DIR (Runpod prefill and/or previous downloads)"
else
    echo "HF cache dir absent: $RUNPOD_MODEL_CACHE_DIR — set the endpoint's Model field to a Hugging Face repo to use Runpod's model store"
fi

# Ollama's own store: everything it pulls lands here, including plain Ollama
# library models and gated hf.co pulls, not just model-store models.
if [ -z "$OLLAMA_MODELS" ]; then
    if [ "$VOLUME_STATE" = "writable" ]; then
        export OLLAMA_MODELS="$VOLUME_ROOT/ollama/models"
    else
        export OLLAMA_MODELS="/root/.ollama/models"
    fi
fi
mkdir -p "$OLLAMA_MODELS" 2>/dev/null || echo "WARN: cannot create $OLLAMA_MODELS"

# Fallback downloads use the same layout as the model store, so on a writable
# volume they persist and the same resolver finds them on the next cold start.
if [ -z "$HUGGINGFACE_HUB_CACHE" ]; then
    if [ "$VOLUME_STATE" = "writable" ]; then
        export HUGGINGFACE_HUB_CACHE="$VOLUME_ROOT/huggingface-cache/hub"
    else
        export HUGGINGFACE_HUB_CACHE="/root/.cache/huggingface/hub"
    fi
fi
export HF_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
mkdir -p "$HUGGINGFACE_HUB_CACHE" 2>/dev/null || true

echo "Ollama store: $OLLAMA_MODELS"
echo "HF cache — read: $RUNPOD_MODEL_CACHE_DIR | write: $HUGGINGFACE_HUB_CACHE"

ollama serve &

echo "Waiting for Ollama server to be ready..."
until curl -sf http://127.0.0.1:11434/api/version > /dev/null; do
    sleep 0.5
done
echo "Ollama server is ready"

if [ -n "$HF_MODEL" ]; then
    echo "Preparing Hugging Face model: $HF_MODEL (HF_MODEL takes precedence — OLLAMA_MODEL='$OLLAMA_MODEL' is ignored)"
elif [ -n "$OLLAMA_MODEL" ]; then
    echo "Preparing Ollama model: $OLLAMA_MODEL"
else
    echo "Neither HF_MODEL nor OLLAMA_MODEL is set — falling back to the worker's default Ollama model"
fi

# Always runs: resolve_default_model has a fallback, so there is always a model to
# prepare. ensure_default_model resolves HF_MODEL over OLLAMA_MODEL, reads Runpod's
# model store first, and uses HF_TOKEN for gated repos. warm_default_model then
# loads the weights into GPU memory: registering only writes the model to disk,
# and without the warm-up the *first user request* pays the multi-minute VRAM
# load and can blow the execution timeout. Warm-up failure is non-fatal (the
# handler retries the load on the first request) and logs its own cause.
if /opt/venv/bin/python -c 'import handler; print("Model ready:", handler.ensure_default_model()); handler.warm_default_model()'; then
    :
else
    echo "WARN: startup model preparation failed (cause in the traceback above) — the handler will retry on the first request, which will be slow and may time out"
fi

cd /
exec /opt/venv/bin/python -u /handler.py
