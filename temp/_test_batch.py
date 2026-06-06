"""Teste do lote. Uso: python temp/_test_batch.py [notex|tex]
  notex: 2 imagens boas + 1 corrompida, SEM textura (rápido) — valida fase 1, resiliência, zip.
  tex:   1 imagem, COM textura (lento) — valida fase 2 (paint + export modes + galeria)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import app

ex = os.path.join("TripoSR", "examples")
good = [os.path.join(ex, "chair.png"), os.path.join(ex, "teapot.png")]
bad = os.path.join("temp", "_corrupt.png")
with open(bad, "wb") as f:
    f.write(b"isto nao e uma imagem")

mode = sys.argv[1] if len(sys.argv) > 1 else "notex"
do_tex = (mode == "tex")
files = [good[0]] if do_tex else (good + [bad])

print(f"[BATCH-TEST] mode={mode} files={[os.path.basename(x) for x in files]}", flush=True)
gen = app.run_batch(
    files, "",                       # files, folder
    app.ENGINE_FAST,                 # engine (TripoSR p/ rapidez)
    app.HQ_DEFAULT_PRESET, 0, 5.0,   # hq_preset, octree, guidance
    app.SMOOTH_DEFAULT,              # smooth
    True, 0.85, 192,                 # do_remove_bg, fg_ratio, mc_res (192 p/ rapidez)
    "glb",                           # export_format
    do_tex, "1024 (seguro)", 60000,  # do_texture, tex_res, tex_max_faces
    "Modelo + textura",              # export_mode
)
last = None
for out in gen:
    assert isinstance(out, tuple) and len(out) == 5, f"yield arity != 5: {len(out)}"
    last = out
    print("  STATUS:", out[1], "| zip:", bool(out[3]), "| gal:", len(out[2] or []), flush=True)

log, status, gallery, zipp, viewer = last
print("\n[BATCH-TEST] FINAL:", status, flush=True)
print("[BATCH-TEST] zip:", zipp, "existe:", bool(zipp) and os.path.exists(zipp), flush=True)
print("[BATCH-TEST] viewer:", viewer, "existe:", bool(viewer) and os.path.exists(viewer), flush=True)
print("[BATCH-TEST] galeria:", len(gallery or []), flush=True)
print("[BATCH-TEST] DONE", flush=True)
