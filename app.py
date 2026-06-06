import os
import sys
import gc
import time
import json
import zipfile
import datetime
import traceback

# Console Windows usa cp1252 e quebra com emojis nos status. Forca UTF-8.
for _stream in ("stdout", "stderr"):
    try:
        getattr(sys, _stream).reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# =============================
# PATHS / SYS.PATH
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPOSR_DIR = os.path.join(BASE_DIR, "TripoSR")
HUNYUAN_DIR = os.path.join(BASE_DIR, "Hunyuan3D2")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Tornar os pacotes `tsr` (TripoSR) e `hy3dgen` (Hunyuan3D) importaveis sem instalar.
if TRIPOSR_DIR not in sys.path:
    sys.path.insert(0, TRIPOSR_DIR)
if os.path.isdir(HUNYUAN_DIR) and HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, HUNYUAN_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cache do HuggingFace dentro do projeto (tanto var antiga quanto a nova).
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", MODEL_DIR)
os.environ.setdefault("HF_HOME", MODEL_DIR)
# rembg/U2Net: usar o modelo local em models/ e nunca a GPU (onnxruntime CPU).
os.environ.setdefault("U2NET_HOME", MODEL_DIR)
# Hunyuan3D: smart_load_model procura primeiro em $HY3DGEN_MODELS/<repo>/<subfolder>.
# Apontando para um diretorio local fixo, ele carrega offline e nunca trava
# esperando lock/download do huggingface_hub.
HY3D_DIR = os.path.join(MODEL_DIR, "hy3d")
os.environ.setdefault("HY3DGEN_MODELS", HY3D_DIR)


def _enable_torch_cudnn_dlls():
    """No Windows o cuDNN 9 do PyTorch carrega seus plugins via LoadLibrary,
    que busca apenas no PATH (ignora add_dll_directory). Sem isso o forward
    na GPU quebra com 'Could not locate cudnn_*64_9.dll'. Precisa rodar ANTES
    de `import torch`.

    NÃO usar glob aqui: o caminho do projeto pode conter colchetes (ex.:
    '[SAAS]') que glob interpreta como classe de caracteres."""
    bases = {
        os.path.dirname(os.path.dirname(sys.executable)),  # raiz do venv
        sys.prefix,
    }
    for base in bases:
        for libdir in ("Lib", "lib"):
            d = os.path.join(base, libdir, "site-packages", "torch", "lib")
            if os.path.isdir(d):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(d)
                except (AttributeError, OSError):
                    pass


if os.name == "nt":
    _enable_torch_cudnn_dlls()

import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps

# Geração de textura (Hunyuan3D-Paint). Import leve: o hy3dgen.texgen (que puxa
# o custom_rasterizer CUDA) é carregado de forma preguiçosa lá dentro, então o
# app abre normalmente mesmo se a textura ainda não estiver disponível.
import texture_gen as texgen

# =============================
# DEVICE
# =============================
def get_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_name(0)
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"

DEVICE = get_device()

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print(f"[INFO] Rodando em: {DEVICE}")

# Motores disponiveis
ENGINE_FAST = "Rápido — TripoSR (~5s)"
ENGINE_HQ = "Alta qualidade — Hunyuan3D-2mini"
ENGINES = [ENGINE_FAST, ENGINE_HQ]

# Presets de qualidade para o motor Hunyuan.
# Cada preset: (subfolder, use_safetensors, steps, octree, num_chunks)
HQ_PRESETS = {
    "Equilíbrio (~1 min)": ("hunyuan3d-dit-v2-mini-turbo", True, 5, 256, 3000),
    "Alta (~1-2 min)": ("hunyuan3d-dit-v2-mini-turbo", True, 5, 384, 2000),
    "Máxima (~4-6 min)": ("hunyuan3d-dit-v2-mini", False, 30, 512, 1500),
    "Extrema (~8-10 min)": ("hunyuan3d-dit-v2-mini", False, 50, 512, 1200),
}
HQ_PRESET_NAMES = list(HQ_PRESETS.keys())
HQ_DEFAULT_PRESET = "Máxima (~4-6 min)"

# =============================
# ESTADO GLOBAL
# =============================
tripo_model = None         # instancia TripoSR
hunyuan_pipe = None        # pipeline Hunyuan3D (atual)
hunyuan_subfolder = None   # qual variante esta carregada
rembg_session = None
MC_BACKEND = "?"

# Guarda o resultado da última geração para a etapa (opcional) de textura.
#   mesh     : trimesh gerado (frame nativo do motor — o que o paint espera)
#   rgba     : PIL RGBA da imagem de entrada com fundo removido (entrada do paint)
#   engine   : "tripo" | "hunyuan"
#   textured : trimesh já texturizado (preenchido após "Gerar textura")
LAST = {"mesh": None, "rgba": None, "engine": None, "textured": None}

# =============================
# PROGRESSO / TEMPO / ETA (log ao vivo + relógio do lote)
# =============================
PROGRESS = {
    "active": False,     # True enquanto um lote roda
    "t0": 0.0,           # início (time.time())
    "done_units": 0,     # unidades concluídas (1 geometria + 1 textura por imagem)
    "total_units": 0,    # total de unidades do lote
    "label": "",         # etapa atual (texto curto)
}


def _fmt_mmss(seconds) -> str:
    try:
        s = int(max(0, round(seconds)))
    except (TypeError, ValueError):
        return "--:--"
    return f"{s // 60:02d}:{s % 60:02d}"


def _ts_line(msg: str) -> str:
    return f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"


def _eta_seconds():
    """ETA = média de seg/unidade concluída × unidades restantes."""
    if not PROGRESS["active"] or PROGRESS["done_units"] <= 0:
        return None
    elapsed = time.time() - PROGRESS["t0"]
    per_unit = elapsed / PROGRESS["done_units"]
    remaining = max(0, PROGRESS["total_units"] - PROGRESS["done_units"])
    return per_unit * remaining


def _clock_text() -> str:
    if not PROGRESS["active"]:
        return ""
    elapsed = time.time() - PROGRESS["t0"]
    eta = _eta_seconds()
    eta_txt = _fmt_mmss(eta) if eta is not None else "calculando..."
    parts = [f"⏱️ Decorrido {_fmt_mmss(elapsed)}", f"⏳ Restante ~{eta_txt}"]
    if PROGRESS["total_units"]:
        parts.append(f"📦 {PROGRESS['done_units']}/{PROGRESS['total_units']} etapas")
    if PROGRESS["label"]:
        parts.append(PROGRESS["label"])
    return "  |  ".join(parts)


def _tick_clock() -> str:
    """Handler do gr.Timer (1s): só lê PROGRESS e formata. Escreve apenas no
    campo 'clock'; vazio quando ocioso."""
    return _clock_text()


class _SubProgress:
    """Mapeia o progress(frac, desc) dos geradores _gen_* para uma sub-faixa
    [base, base+span] da barra real do Gradio e espelha o desc em
    PROGRESS['label']. Permite reaproveitar _gen_tripo/_gen_hunyuan SEM alterá-los.
    Compatível com gr.Progress (callable + .tqdm)."""
    def __init__(self, real_progress, base=0.0, span=1.0, prefix=""):
        self._real = real_progress
        self._base = base
        self._span = span
        self._prefix = prefix

    def __call__(self, frac=None, desc=None, total=None, **kwargs):
        if desc is not None:
            PROGRESS["label"] = f"{self._prefix}{desc}"
        if self._real is not None and frac is not None:
            try:
                f = self._base + self._span * float(frac)
                self._real(min(max(f, 0.0), 1.0),
                           desc=(f"{self._prefix}{desc}" if desc else None))
            except Exception:
                pass

    def tqdm(self, iterable, *args, **kwargs):
        return iterable


# =============================
# LOAD MODELS
# =============================
def load_tripo():
    global tripo_model, MC_BACKEND
    if tripo_model is not None:
        return tripo_model
    from tsr.system import TSR
    from tsr.models.isosurface import get_marching_cubes_backend

    print("[INFO] Carregando TripoSR...")
    m = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    m.renderer.set_chunk_size(8192)
    m.to(DEVICE)
    m.eval()
    tripo_model = m
    MC_BACKEND = get_marching_cubes_backend()
    print(f"[INFO] TripoSR carregado. MC backend: {MC_BACKEND}")
    return tripo_model


def _ensure_hunyuan_local(subfolder: str):
    """Garante o modelo local (layout que smart_load_model espera) com
    download resumível e único. Assim from_pretrained carrega offline."""
    repo = "tencent/Hunyuan3D-2mini"
    local_repo = os.path.join(HY3D_DIR, repo.replace("/", os.sep))
    needed = os.path.join(local_repo, subfolder, "config.yaml")
    if not os.path.exists(needed):
        from huggingface_hub import snapshot_download
        print(f"[INFO] Baixando {repo}/{subfolder} (uma vez, ~3.8GB)...")
        os.makedirs(local_repo, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            allow_patterns=[f"{subfolder}/*"],
            local_dir=local_repo,
            max_workers=4,
        )


def load_hunyuan(subfolder: str = "hunyuan3d-dit-v2-mini-turbo",
                  use_safetensors: bool = True):
    """Carrega (ou troca) a variante do pipeline Hunyuan. Só uma variante
    fica residente por vez (cabe melhor nos 4 GB)."""
    global hunyuan_pipe, hunyuan_subfolder
    if hunyuan_pipe is not None and hunyuan_subfolder == subfolder:
        return hunyuan_pipe
    if not os.path.isdir(HUNYUAN_DIR):
        raise RuntimeError("Hunyuan3D2 não encontrado. Rode install.bat novamente.")
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    # libera a variante anterior, se houver
    if hunyuan_pipe is not None:
        del hunyuan_pipe
        hunyuan_pipe = None
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    _ensure_hunyuan_local(subfolder)
    print(f"[INFO] Carregando Hunyuan3D {subfolder}...")
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder=subfolder,
        use_safetensors=use_safetensors,
        variant="fp16",
        dtype=torch.float16,
        device=DEVICE,
    )
    pipe.enable_flashvdm(enabled=True)
    hunyuan_pipe = pipe
    hunyuan_subfolder = subfolder
    print(f"[INFO] Hunyuan3D {subfolder} carregado (flashvdm).")
    return hunyuan_pipe


def load_model(engine: str = ENGINE_FAST, hq_preset: str = HQ_DEFAULT_PRESET):
    """Carrega o motor escolhido. Retorna mensagem de status."""
    try:
        if engine == ENGINE_HQ:
            sub, safet, _steps, _oct, _nc = HQ_PRESETS[hq_preset]
            load_hunyuan(sub, safet)
            return f"✅ Hunyuan3D pronto [{hq_preset}] | Device: {DEVICE}"
        load_tripo()
        return f"✅ TripoSR pronto | MC: {MC_BACKEND} | Device: {DEVICE}"
    except Exception:
        return f"❌ Erro ao carregar modelo:\n{traceback.format_exc()}"


def _get_rembg_session():
    global rembg_session
    if rembg_session is None:
        import rembg
        # Forca CPU: onnxruntime-gpu exige cuDNN 9 (ausente) e quebra.
        rembg_session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
    return rembg_session


def _remove_bg_rgba(img: Image.Image, foreground_ratio: float) -> Image.Image:
    """Remove fundo e devolve RGBA recortado (formato que o Hunyuan espera)."""
    from tsr.utils import remove_background, resize_foreground

    img = ImageOps.exif_transpose(img)
    img = remove_background(img, _get_rembg_session())
    img = resize_foreground(img, foreground_ratio)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def _preprocess(img: Image.Image, foreground_ratio: float, do_remove_bg: bool) -> Image.Image:
    """Aplica EXIF, remove fundo, compõe em cinza (igual run.py oficial)."""
    from tsr.utils import remove_background, resize_foreground

    img = ImageOps.exif_transpose(img)

    if do_remove_bg:
        img = remove_background(img, _get_rembg_session())
        img = resize_foreground(img, foreground_ratio)
        arr = np.array(img).astype(np.float32) / 255.0
        # composite contra cinza 0.5 (distribuição usada no treino)
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")

    return img


SMOOTH_LEVELS = {
    "Desligado": 0,
    "Suave": 8,
    "Forte (recomendado p/ máx. qualidade)": 18,
    "Máximo": 35,
}
SMOOTH_NAMES = list(SMOOTH_LEVELS.keys())
SMOOTH_DEFAULT = "Forte (recomendado p/ máx. qualidade)"


def _refine_mesh(mesh, smooth_level: str):
    """Remove o 'quadriculado' do marching cubes via suavização Taubin
    (lambda/mu — NÃO encolhe o volume, preserva características).
    Pura (trimesh+numpy), sem compilação."""
    iters = SMOOTH_LEVELS.get(smooth_level, 0)
    if iters <= 0 or mesh is None or len(mesh.vertices) == 0:
        return mesh
    import trimesh

    # Limpeza geométrica (API trimesh 4.x) — opcional, não pode impedir o smooth.
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass

    # Suavização Taubin (lambda/mu) — NÃO encolhe volume, mata o staircase.
    try:
        trimesh.smoothing.filter_taubin(
            mesh, lamb=0.5, nu=0.53, iterations=int(iters)
        )
    except Exception:
        print("[WARN] suavização Taubin falhou:\n" + traceback.format_exc())
    return mesh


def _export(mesh, export_format: str):
    ext = export_format.lower().lstrip(".")
    if ext not in ("glb", "obj", "ply"):
        ext = "glb"
    filename = f"mesh_{time.time_ns()}.{ext}"
    path = os.path.join(OUTPUT_DIR, filename)
    mesh.export(path)
    return path, filename


def _gen_tripo(img, do_remove_bg, foreground_ratio, mc_resolution, export_format, smooth, progress):
    from tsr.utils import to_gradio_3d_orientation

    progress(0.1, desc="Pré-processando imagem...")
    proc_img = _preprocess(img, foreground_ratio, do_remove_bg)
    m = load_tripo()

    progress(0.35, desc="TripoSR: gerando código de cena...")
    with torch.inference_mode():
        scene_codes = m(proc_img, device=DEVICE)

    progress(0.7, desc=f"TripoSR: extraindo mesh ({mc_resolution})...")
    with torch.inference_mode():
        meshes = m.extract_mesh(scene_codes, has_vertex_color=True, resolution=int(mc_resolution))
    mesh = to_gradio_3d_orientation(meshes[0])

    progress(0.9, desc="Suavizando malha...")
    mesh = _refine_mesh(mesh, smooth)

    LAST["mesh"] = mesh          # guardado p/ a etapa opcional de textura
    LAST["engine"] = "tripo"

    progress(0.95, desc="Exportando...")
    path, filename = _export(mesh, export_format)
    return path, filename, len(mesh.vertices), len(mesh.faces)


def _gen_hunyuan(img, foreground_ratio, export_format, hq_preset, octree_override, smooth, guidance_scale, progress):
    sub, safet, steps, octree, num_chunks = HQ_PRESETS[hq_preset]
    if octree_override and int(octree_override) > 0:
        octree = int(octree_override)
        # mais resolucao -> menos chunks para conter o pico de VRAM
        num_chunks = max(800, int(num_chunks * (512 / max(octree, 1))))

    progress(0.05, desc="Pré-processando (remoção de fundo)...")
    rgba = _remove_bg_rgba(img, foreground_ratio)

    progress(0.15, desc=f"Carregando modelo Hunyuan ({hq_preset})...")
    pipe = load_hunyuan(sub, safet)

    progress(0.4, desc=f"Hunyuan3D: gerando geometria ({steps} passos, octree {octree})...")
    # O pico de VRAM depende de octree_resolution/num_chunks (NAO do tamanho
    # da imagem), entao estes valores mantem o pico estavel p/ qualquer entrada.
    with torch.inference_mode():
        mesh = pipe(
            image=rgba,
            num_inference_steps=steps,
            octree_resolution=octree,
            num_chunks=num_chunks,
            mc_algo="mc",
            guidance_scale=float(guidance_scale),
        )[0]

    progress(0.88, desc="Suavizando malha (anti-quadriculado)...")
    mesh = _refine_mesh(mesh, smooth)

    LAST["mesh"] = mesh          # frame nativo do Hunyuan = o que o paint espera
    LAST["engine"] = "hunyuan"

    progress(0.92, desc="Exportando...")
    path, filename = _export(mesh, export_format)
    return path, filename, len(mesh.vertices), len(mesh.faces)


# =============================
# GENERATE 3D
# =============================
def generate_3d(
    img,
    engine: str,
    hq_preset: str,
    octree_override: int,
    smooth: str,
    do_remove_bg: bool,
    foreground_ratio: float,
    mc_resolution: int,
    export_format: str,
    guidance_scale: float = 5.0,
    progress=gr.Progress(),
):
    if img is None:
        return None, None, "⚠️ Envie uma imagem"

    t0 = time.time()
    note = ""
    try:
        if engine == ENGINE_HQ:
            try:
                path, fn, nv, nf = _gen_hunyuan(
                    img, foreground_ratio, export_format, hq_preset, octree_override, smooth, guidance_scale, progress
                )
            except Exception:
                # Fallback automatico: nunca deixa o usuario sem resultado.
                print("[WARN] Hunyuan falhou, fallback TripoSR:\n" + traceback.format_exc())
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                note = " (⚠️ Hunyuan falhou — usei TripoSR como fallback)"
                path, fn, nv, nf = _gen_tripo(
                    img, do_remove_bg, foreground_ratio, mc_resolution, export_format, smooth, progress
                )
        else:
            path, fn, nv, nf = _gen_tripo(
                img, do_remove_bg, foreground_ratio, mc_resolution, export_format, smooth, progress
            )

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # prepara entrada da textura (independe do frame do mesh) e zera textura antiga
        try:
            LAST["rgba"] = _remove_bg_rgba(img, foreground_ratio)
        except Exception:
            LAST["rgba"] = None
        LAST["textured"] = None

        elapsed = time.time() - t0
        return path, path, (
            f"✅ 3D gerado em {elapsed:.1f}s | {nv} vértices, {nf} faces | {fn}{note}"
            "  →  agora você pode 🎨 Gerar textura."
        )

    except Exception:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return None, None, f"❌ Erro:\n{traceback.format_exc()}"


def preview_no_bg(img, foreground_ratio: float):
    if img is None:
        return None, "Envie uma imagem"
    try:
        out = _preprocess(img, foreground_ratio, True)
        return out, "✅ Fundo removido"
    except Exception:
        return None, f"❌ {traceback.format_exc()}"


# =============================
# TEXTURA (Hunyuan3D-Paint)
# =============================
EXPORT_MODE_MAP = {
    "Modelo + textura": "both",
    "Somente textura": "texture",
    "Somente modelo": "model",
}


def _parse_tex_res(choice) -> int:
    s = str(choice)
    if "2048" in s:
        return 2048
    if "1536" in s:
        return 1536
    return 1024


def _free_generator_models():
    """Em 4 GB não cabem o modelo de geração + o de textura ao mesmo tempo.
    Antes de texturizar, descarrega TripoSR/Hunyuan da memória (recarregam
    sob demanda na próxima geração)."""
    global tripo_model, hunyuan_pipe, hunyuan_subfolder
    tripo_model = None
    hunyuan_pipe = None
    hunyuan_subfolder = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def generate_texture(tex_res_choice: str, tex_max_faces: int,
                     export_format: str, progress=gr.Progress()):
    """Repinta o último modelo gerado a partir da imagem de entrada."""
    if LAST.get("mesh") is None:
        return None, None, "⚠️ Gere um modelo 3D primeiro, depois clique em 🎨 Gerar textura."
    if LAST.get("rgba") is None:
        return None, None, "⚠️ Imagem de origem indisponível — gere o modelo novamente."

    res = _parse_tex_res(tex_res_choice)
    t0 = time.time()
    try:
        _free_generator_models()  # abre espaço na VRAM para o paint
        textured = texgen.texture_mesh(
            LAST["mesh"], LAST["rgba"],
            subfolder="hunyuan3d-paint-v2-0-turbo",
            render_size=res, texture_size=res,
            max_faces=int(tex_max_faces), progress=progress,
        )
        LAST["textured"] = textured

        progress(0.98, desc="Gerando preview texturizado (GLB)...")
        # o visualizador 3D precisa de um GLB com a textura embutida
        glb_path, _ = texgen.export_result(textured, LAST["mesh"], "both", "glb", OUTPUT_DIR)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elapsed = time.time() - t0
        return glb_path, glb_path, (
            f"✅ Textura gerada em {elapsed:.1f}s @ {res}px | "
            "use 💾 Baixar para escolher modelo / textura / ambos."
        )
    except Exception:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        return None, None, f"❌ Erro ao gerar textura:\n{traceback.format_exc()}"


def export_download(export_mode_choice: str, export_format: str):
    """Exporta o resultado conforme o modo escolhido (sem reprocessar nada)."""
    mode = EXPORT_MODE_MAP.get(export_mode_choice, "both")
    if LAST.get("mesh") is None:
        return None, "⚠️ Gere um modelo primeiro."
    if mode in ("texture", "both") and LAST.get("textured") is None:
        return None, "⚠️ Esse modo precisa de textura — clique em 🎨 Gerar textura antes."
    try:
        path, fn = texgen.export_result(
            LAST.get("textured"), LAST.get("mesh"), mode, export_format, OUTPUT_DIR)
        return path, f"✅ Exportado ({export_mode_choice}): {fn}"
    except Exception:
        return None, f"❌ {traceback.format_exc()}"


# =============================
# LOTE (BATCH) — várias imagens, uma a uma
# =============================
_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _collect_batch_inputs(files, folder_path):
    """Une os arquivos enviados + (opcional) uma pasta. Usa os.walk/scandir —
    NUNCA glob (o caminho do projeto pode conter colchetes)."""
    paths = []
    if files:
        for f in files:
            p = getattr(f, "name", None) or (f if isinstance(f, str) else None)
            if p and str(p).lower().endswith(_IMG_EXTS):
                paths.append(str(p))
    if folder_path and str(folder_path).strip():
        fp = str(folder_path).strip().strip('"')
        if os.path.isdir(fp):
            for root, _dirs, names in os.walk(fp):
                for n in names:
                    if n.lower().endswith(_IMG_EXTS):
                        paths.append(os.path.join(root, n))
    seen, uniq = set(), []
    for p in sorted(paths):
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            uniq.append(p)
    return uniq


def _zip_dir(folder):
    zip_path = folder.rstrip("\\/") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, names in os.walk(folder):
            for n in names:
                full = os.path.join(root, n)
                z.write(full, os.path.relpath(full, folder))
    return zip_path


def _write_manifest(folder, results):
    try:
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception:
        print("[WARN] manifest:\n" + traceback.format_exc())


def run_batch(files, folder_path,
              engine, hq_preset, octree_override, guidance_scale, smooth,
              do_remove_bg, foreground_ratio, mc_resolution, export_format,
              do_texture, tex_res, tex_max_faces, export_mode_choice,
              progress=gr.Progress()):
    """Processa N imagens em 2 fases (geometria de todas → textura de todas),
    com tolerância a falha por imagem e saída incremental. GERADOR: dá yield de
    (log, status, galeria, zip, viewer) a cada etapa."""
    log_lines, gallery = [], []
    L = log_lines.append

    imgs = _collect_batch_inputs(files, folder_path)
    if not imgs:
        yield ("", "⚠️ Nenhuma imagem encontrada. Envie arquivos ou informe uma pasta válida.",
               [], None, None)
        return

    snapshot = dict(LAST)                 # preserva o estado do fluxo de imagem única
    n = len(imgs)
    res = _parse_tex_res(tex_res)
    export_mode = EXPORT_MODE_MAP.get(export_mode_choice, "both")
    PROGRESS.update(active=True, t0=time.time(), done_units=0,
                    total_units=n + (n if do_texture else 0), label="iniciando")

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = os.path.join(OUTPUT_DIR, f"batch_{ts}")
    os.makedirs(out_root, exist_ok=True)

    L(_ts_line(f"🚀 Lote: {n} imagem(ns) | motor={engine} | textura={'sim' if do_texture else 'não'}"))
    L(_ts_line(f"📁 Saída: {out_root}"))
    last_viewer = None
    items = []
    yield ("\n".join(log_lines), f"Iniciando lote ({n} imagens)...", gallery, None, None)

    try:
        # ---------------- FASE 1: GEOMETRIA ----------------
        for i, src in enumerate(imgs):
            stem = os.path.splitext(os.path.basename(src))[0]
            item_dir = os.path.join(out_root, f"{i+1:03d}_{stem}")
            os.makedirs(item_dir, exist_ok=True)
            entry = {"idx": i + 1, "src": src, "dir": item_dir, "stem": stem,
                     "mesh_path": None, "rgba_path": None, "ok": False, "error": None,
                     "model_file": None, "textured_file": None, "texture_file": None}
            items.append(entry)
            PROGRESS["label"] = f"[{i+1}/{n}] geometria"
            L(_ts_line(f"🖼️ [{i+1}/{n}] {os.path.basename(src)} — geometria..."))
            yield ("\n".join(log_lines), f"Fase 1 (geometria): {i+1}/{n}", gallery, None, last_viewer)
            try:
                img = Image.open(src)
                sub = _SubProgress(progress, i / PROGRESS["total_units"],
                                   1.0 / PROGRESS["total_units"], prefix=f"[{i+1}/{n}] ")
                if engine == ENGINE_HQ:
                    try:
                        gpath, _, _, _ = _gen_hunyuan(img, foreground_ratio, "ply", hq_preset,
                                                      octree_override, smooth, guidance_scale, sub)
                    except Exception:
                        print("[WARN] Hunyuan falhou no lote, fallback TripoSR:\n" + traceback.format_exc())
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                        L(_ts_line("   ⚠️ Hunyuan falhou — usando TripoSR como fallback"))
                        gpath, _, _, _ = _gen_tripo(img, do_remove_bg, foreground_ratio,
                                                    mc_resolution, "ply", smooth, sub)
                else:
                    gpath, _, _, _ = _gen_tripo(img, do_remove_bg, foreground_ratio,
                                                mc_resolution, "ply", smooth, sub)
                # remove o arquivo solto que _gen escreveu em OUTPUT_DIR (reexporto no item_dir)
                try:
                    if gpath and os.path.exists(gpath):
                        os.remove(gpath)
                except Exception:
                    pass

                mesh = LAST["mesh"]
                mesh_path = os.path.join(item_dir, "mesh.ply")
                mesh.export(mesh_path)
                rgba_path = os.path.join(item_dir, "input_rgba.png")
                try:
                    _remove_bg_rgba(img, foreground_ratio).save(rgba_path)
                except Exception:
                    rgba_path = None
                mp, _ = texgen.export_result(None, mesh, "model", export_format, item_dir)
                entry.update(mesh_path=mesh_path, rgba_path=rgba_path, ok=True, model_file=mp)
                # glb p/ o visualizador
                if str(mp).lower().endswith(".glb"):
                    last_viewer = mp
                else:
                    try:
                        last_viewer, _ = texgen.export_result(None, mesh, "model", "glb", item_dir)
                    except Exception:
                        pass
                L(_ts_line(f"   ✅ geometria: {len(mesh.vertices)} v / {len(mesh.faces)} f → {os.path.basename(mp)}"))
            except Exception:
                entry["error"] = traceback.format_exc()
                L(_ts_line(f"   ❌ erro na geometria de {os.path.basename(src)} (pulando)"))
                print(f"[WARN] geometria {src}:\n" + entry["error"])
            finally:
                LAST["mesh"] = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                PROGRESS["done_units"] += 1
                yield ("\n".join(log_lines), f"Fase 1: {i+1}/{n}", gallery, None, last_viewer)

        # ---------------- LIBERA MODELOS ----------------
        L(_ts_line("🧹 Liberando modelos de geração da VRAM..."))
        yield ("\n".join(log_lines), "Liberando modelos da VRAM...", gallery, None, last_viewer)
        _free_generator_models()

        # ---------------- FASE 2: TEXTURA ----------------
        if do_texture:
            ok_items = [e for e in items if e["ok"] and e["mesh_path"] and e["rgba_path"]]
            PROGRESS["total_units"] = n + len(ok_items)   # ETA exata após saber os sucessos
            for j, entry in enumerate(ok_items):
                i = entry["idx"]
                PROGRESS["label"] = f"[{i}/{n}] textura"
                L(_ts_line(f"🎨 [{i}/{n}] {entry['stem']} — textura @{res}px (lento em 4 GB)..."))
                yield ("\n".join(log_lines), f"Fase 2 (textura): {j+1}/{len(ok_items)}", gallery, None, last_viewer)
                try:
                    sub = _SubProgress(progress, (n + j) / PROGRESS["total_units"],
                                       1.0 / PROGRESS["total_units"], prefix=f"[{i}/{n}] ")
                    textured = texgen.texture_mesh_from_files(
                        entry["mesh_path"], entry["rgba_path"],
                        subfolder="hunyuan3d-paint-v2-0-turbo",
                        render_size=res, texture_size=res,
                        max_faces=int(tex_max_faces), progress=sub)
                    tp, _ = texgen.export_result(textured, None, export_mode, export_format, entry["dir"])
                    entry["textured_file"] = tp
                    # viewer: reusa o arquivo entregue se já for um glb texturizado;
                    # senão gera um glb só para o visualizador (evita duplicata)
                    if export_mode == "both" and str(tp).lower().endswith(".glb"):
                        last_viewer = tp
                    else:
                        try:
                            last_viewer, _ = texgen.export_result(textured, None, "both", "glb", entry["dir"])
                        except Exception:
                            pass
                    timg = texgen.get_texture_image(textured)
                    if timg is not None:
                        tex_png = os.path.join(entry["dir"], "textura.png")
                        timg.save(tex_png)
                        entry["texture_file"] = tex_png
                        gallery = gallery + [(tex_png, f"[{i}] {entry['stem']}")]
                    L(_ts_line(f"   ✅ textura → {os.path.basename(tp)}"))
                except Exception:
                    entry["error"] = (entry.get("error") or "") + "\nTEXTURA:\n" + traceback.format_exc()
                    L(_ts_line(f"   ❌ erro na textura de {entry['stem']} (mantida só a geometria)"))
                    print(f"[WARN] textura {entry['stem']}:\n" + traceback.format_exc())
                finally:
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    PROGRESS["done_units"] += 1
                    yield ("\n".join(log_lines), f"Fase 2: {j+1}/{len(ok_items)}", gallery, None, last_viewer)
            if export_mode in ("texture", "both") and export_format == "ply":
                L(_ts_line("ℹ️ .ply não embute textura — para modelo+textura use glb ou obj."))

        # ---------------- FINALIZA ----------------
        for e in items:
            results_entry = {k: e.get(k) for k in
                             ("idx", "src", "ok", "model_file", "textured_file", "texture_file")}
            results_entry["error"] = bool(e.get("error"))
            e["_manifest"] = results_entry
        _write_manifest(out_root, [e["_manifest"] for e in items])
        try:
            with open(os.path.join(out_root, "log.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(log_lines))
        except Exception:
            pass
        zip_path = None
        try:
            zip_path = _zip_dir(out_root)
        except Exception:
            L(_ts_line("⚠️ Falha ao criar o .zip (os arquivos estão na pasta de saída)."))
        n_ok = sum(1 for e in items if e["ok"])
        n_tex = sum(1 for e in items if e.get("textured_file"))
        elapsed = time.time() - PROGRESS["t0"]
        L(_ts_line(f"🏁 Concluído em {_fmt_mmss(elapsed)} | geometria {n_ok}/{n}"
                   + (f" | textura {n_tex}/{n}" if do_texture else "")))
        final = (f"✅ Lote concluído: {n_ok}/{n} modelos"
                 + (f", {n_tex}/{n} texturas" if do_texture else "")
                 + f" em {_fmt_mmss(elapsed)}")
        yield ("\n".join(log_lines), final, gallery, zip_path, last_viewer)
    except Exception:
        L(_ts_line("❌ Erro fatal no lote:\n" + traceback.format_exc()))
        yield ("\n".join(log_lines), "❌ Erro no lote (veja o log)", gallery, None, last_viewer)
    finally:
        PROGRESS["active"] = False
        LAST.update(snapshot)
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# =============================
# UI
# =============================
def build_ui():
    with gr.Blocks(title="3D Generator PRO", analytics_enabled=False) as app:
        gr.Markdown("# 🚀 Gerador de Modelo 3D — imagem → 3D + textura")
        gr.Markdown(f"**Device:** `{DEVICE}`  |  Cache: `{MODEL_DIR}`")

        with gr.Row():
            status = gr.Textbox(
                label="Status", value="Pronto. Ajuste tudo na aba ⚙️ Configurações.",
                interactive=False, scale=3,
            )
            clock = gr.Textbox(label="⏱️ Tempo", value="", interactive=False, scale=2)
        timer = gr.Timer(1.0)

        with gr.Tabs():
            # ===================== ABA: IMAGEM ÚNICA =====================
            with gr.Tab("🖼️ Imagem única"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(type="pil", label="Imagem de entrada", image_mode="RGBA")
                        with gr.Row():
                            preview_btn = gr.Button("👁️ Preview (remover fundo)")
                            run_btn = gr.Button("⚡ Gerar 3D", variant="primary")
                        tex_btn = gr.Button("🎨 Gerar textura (após gerar o 3D)", variant="primary")
                        with gr.Row():
                            export_mode = gr.Radio(
                                list(EXPORT_MODE_MAP.keys()), value="Modelo + textura",
                                label="O que baixar", scale=3,
                            )
                            export_btn = gr.Button("💾 Baixar", scale=1)
                        gr.Markdown("ℹ️ Motor, qualidade, textura e formato ficam na aba **⚙️ Configurações**.")
                    with gr.Column(scale=1):
                        preview_img = gr.Image(label="Preview pós-remoção", type="pil")
                        viewer = gr.Model3D(label="Preview 3D")
                        file_out = gr.File(label="Download")

            # ===================== ABA: LOTE =====================
            with gr.Tab("📚 Lote (várias imagens)"):
                gr.Markdown(
                    "Processa **várias imagens, uma a uma** (geometria de todas → textura de todas). "
                    "Tolera falhas (pula a imagem com problema) e salva incrementalmente em `outputs/batch_…`.  \n"
                    "⚠️ **Com textura em 4 GB, ~10 min por imagem** — dezenas levam horas. "
                    "Desligue a textura para um lote bem mais rápido (só geometria)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            file_count="multiple", file_types=["image"],
                            label="Imagens (arraste várias)",
                        )
                        batch_folder = gr.Textbox(
                            label="...ou caminho de uma pasta (opcional)",
                            placeholder=r"E:\minhas_imagens",
                        )
                        do_texture = gr.Checkbox(value=True, label="Também gerar textura (lento em 4 GB)")
                        batch_export_mode = gr.Radio(
                            list(EXPORT_MODE_MAP.keys()), value="Modelo + textura",
                            label="O que exportar por imagem",
                        )
                        run_batch_btn = gr.Button("🚀 Processar lote", variant="primary")
                        gr.Markdown("Usa motor/qualidade/textura da aba **⚙️ Configurações**.")
                    with gr.Column(scale=1):
                        batch_gallery = gr.Gallery(label="Texturas geradas", columns=3, height=260)
                        batch_viewer = gr.Model3D(label="Último resultado")
                        batch_zip = gr.File(label="📦 Baixar tudo (.zip)")
                batch_log = gr.Textbox(
                    label="Log do lote", lines=16, max_lines=16,
                    interactive=False, autoscroll=True,
                )

            # ===================== ABA: CONFIGURAÇÕES (compartilhada) =====================
            with gr.Tab("⚙️ Configurações"):
                gr.Markdown("Estes ajustes valem para **imagem única e lote**.")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Geometria")
                        engine = gr.Radio(ENGINES, value=ENGINE_HQ, label="Motor de geração")
                        hq_preset = gr.Radio(
                            HQ_PRESET_NAMES, value=HQ_DEFAULT_PRESET,
                            label="Qualidade (só Hunyuan)",
                        )
                        guidance_scale = gr.Slider(
                            1.0, 10.0, value=5.0, step=0.5,
                            label="Guidance scale (Hunyuan; ↑ = mais fiel à imagem)",
                        )
                        smooth = gr.Radio(
                            SMOOTH_NAMES, value=SMOOTH_DEFAULT,
                            label="Suavização (anti-quadriculado)",
                        )
                        octree_override = gr.Slider(
                            0, 768, value=0, step=64,
                            label="Octree (0 = usar o do preset; ↑ = mais detalhe)",
                        )
                        mc_resolution = gr.Slider(
                            64, 384, value=256, step=32,
                            label="Resolução marching cubes (só TripoSR)",
                        )
                        do_remove_bg = gr.Checkbox(value=True, label="Remover fundo automaticamente")
                        foreground_ratio = gr.Slider(0.5, 1.0, value=0.85, step=0.05, label="Foreground ratio")
                        export_format = gr.Radio(["glb", "obj", "ply"], value="glb", label="Formato de export")
                    with gr.Column():
                        gr.Markdown("#### Textura (Hunyuan3D-Paint)")
                        tex_res = gr.Radio(
                            ["1024 (seguro)", "1536", "2048 (máxima, mais lento)"],
                            value="1024 (seguro)", label="Resolução da textura",
                        )
                        tex_max_faces = gr.Slider(
                            40000, 500000, value=120000, step=20000,
                            label="Faces para texturizar (decimação p/ caber em 4 GB; ↓ = mais rápido)",
                        )
                        gr.Markdown("#### Modelo")
                        load_btn = gr.Button("🔄 (Re)carregar modelo selecionado")

        # ---- WIRING (no FIM: todos os componentes já existem) ----
        load_btn.click(load_model, [engine, hq_preset], status)
        preview_btn.click(preview_no_bg, [input_img, foreground_ratio], [preview_img, status])
        run_btn.click(
            generate_3d,
            [input_img, engine, hq_preset, octree_override, smooth,
             do_remove_bg, foreground_ratio, mc_resolution, export_format, guidance_scale],
            [viewer, file_out, status],
        )
        tex_btn.click(generate_texture, [tex_res, tex_max_faces, export_format], [viewer, file_out, status])
        export_btn.click(export_download, [export_mode, export_format], [file_out, status])
        run_batch_btn.click(
            run_batch,
            [batch_files, batch_folder, engine, hq_preset, octree_override, guidance_scale, smooth,
             do_remove_bg, foreground_ratio, mc_resolution, export_format,
             do_texture, tex_res, tex_max_faces, batch_export_mode],
            [batch_log, status, batch_gallery, batch_zip, batch_viewer],
        )
        timer.tick(_tick_clock, None, clock, show_progress="hidden")

    return app


# =============================
# RUN
# =============================
if __name__ == "__main__":
    # Pré-aquece o modelo em background quando há tempo? Aqui carregamos só sob demanda
    # para que a UI abra rápido mesmo se o checkpoint precisar ser baixado.
    app = build_ui()
    app.queue().launch(inbrowser=True, server_name="127.0.0.1")
