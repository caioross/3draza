"""Teste end-to-end da textura: imagem -> mesh -> Hunyuan-Paint -> 3 exports.
Uso: python temp/_test_texture.py [tripo|hunyuan] [res] [maxfaces] [imagem]"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import app  # configura env/sys.path/cudnn; nao lanca (guard __main__)
import torch
import texture_gen as texgen
from PIL import Image

engine = sys.argv[1] if len(sys.argv) > 1 else "tripo"
res = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 80000
imgpath = sys.argv[4] if len(sys.argv) > 4 else os.path.join("TripoSR", "examples", "robot.png")


class Prog:
    """Stub compatível com a assinatura usada (progress(v, desc=...))."""
    def __call__(self, v, desc=""):
        print(f"   [{v:5.0%}] {desc}", flush=True)


prog = Prog()
img = Image.open(imgpath)
print(f"[TEST] imagem={imgpath} engine={engine} res={res} maxfaces={maxf}", flush=True)

t0 = time.time()
if engine == "hunyuan":
    path, fn, nv, nf = app._gen_hunyuan(
        img, 0.85, "glb", "Equilíbrio (~1 min)", 0, app.SMOOTH_DEFAULT, prog)
else:
    path, fn, nv, nf = app._gen_tripo(
        img, True, 0.85, 256, "glb", app.SMOOTH_DEFAULT, prog)
print(f"[TEST] mesh: {nv} verts / {nf} faces em {time.time()-t0:.1f}s ({fn})", flush=True)

rgba = app._remove_bg_rgba(img, 0.85)
mesh = app.LAST["mesh"]

app._free_generator_models()  # libera VRAM (igual ao fluxo do app)
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

t1 = time.time()
try:
    textured = texgen.texture_mesh(
        mesh, rgba, render_size=res, texture_size=res, max_faces=maxf, progress=prog)
except Exception:
    print("[TEST] FALHA na textura:\n" + traceback.format_exc(), flush=True)
    sys.exit(1)

dt = time.time() - t1
peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

gp, gn = texgen.export_result(textured, mesh, "both", "glb", app.OUTPUT_DIR)
tp, tn = texgen.export_result(textured, mesh, "texture", "png", app.OUTPUT_DIR)
mp, mn = texgen.export_result(textured, mesh, "model", "glb", app.OUTPUT_DIR)
ti = texgen.get_texture_image(textured)

print(f"[TEST] textura OK em {dt:.1f}s | VRAM pico {peak:.2f} GB | "
      f"tex={ti.size if ti else None}", flush=True)
print(f"[TEST] exports: both={gn} | texture={tn} | model={mn}", flush=True)
print("[TEST] DONE", flush=True)
