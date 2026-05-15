#!/bin/bash
# Patch RMBG-2.0's birefnet.py if it exists in HF cache.
# The custom model code calls torch.linspace().item() during __init__,
# which fails with meta tensors in PyTorch 2.9.1 + transformers 5.x.
# Fix: add device='cpu' to force real tensor creation inside meta context.

BIREFNET_FILES=$(find /workspace/cache/huggingface -name "birefnet.py" -path "*/briaai/*" 2>/dev/null)
for f in $BIREFNET_FILES; do
    if grep -q "torch\.linspace(0, drop_path_rate, sum(depths))" "$f" && ! grep -q "device='cpu'" "$f"; then
        echo "[ENTRYPOINT] Patching $f — adding device='cpu' to torch.linspace"
        sed -i "s/torch\.linspace(0, drop_path_rate, sum(depths))/torch.linspace(0, drop_path_rate, sum(depths), device='cpu')/g" "$f"
    fi
done

# GLB detection is now handled server-side in the reverse proxy (server.py).
# The old fetch() interceptor approach didn't work in Gradio 6.x because
# model-viewer captures window.fetch before the custom <head> script runs.

# Start Gradio in background (loads model, serves on port 7860)
echo "[ENTRYPOINT] Starting Gradio app on port 7860..."
cd /app/TRELLIS.2
python3 app.py &

# Start FastAPI wrapper in foreground on port 8080
# Shows loading page immediately while Gradio initializes the model
echo "[ENTRYPOINT] Starting wrapper server on port 8080..."
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8080 --log-level info --app-dir /app/trellis2-wrapper
