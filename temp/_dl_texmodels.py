import os, sys, time
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target = os.path.join(BASE, "models", "hy3d", "tencent", "Hunyuan3D-2")
os.makedirs(target, exist_ok=True)

patterns = ["hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"]
t0 = time.time()
for pat in patterns:
    print(f"[DL] baixando {pat} -> {target}", flush=True)
    snapshot_download(
        repo_id="tencent/Hunyuan3D-2",
        allow_patterns=[pat],
        local_dir=target,
        max_workers=4,
    )
    print(f"[DL] ok {pat}  ({time.time()-t0:.0f}s)", flush=True)
print("[DL] DONE", flush=True)
