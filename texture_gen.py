"""
Geração de textura SOTA a partir da imagem, via Hunyuan3D-Paint.

Pega o mesh já gerado (TripoSR ou Hunyuan) + a imagem de entrada e repinta o
modelo com difusão multi-view (delight -> multiview -> back-projection/baking),
produzindo uma textura UV de alta resolução. Funciona até nos lados que a
imagem não mostra.

Otimizado para os 4 GB da RTX 3050:
  - enable_model_cpu_offload: os modelos de difusão saem da VRAM quando ociosos
    (só o módulo ativo fica na GPU) -> pico cabe em ~3-4 GB.
  - resolução de render/textura configurável (1024 padrão; 2048 = máxima).
  - decimação do mesh antes do unwrap/baking (xatlas e o baking crescem com o
    nº de vértices; a malha Hunyuan tem ~1M+ faces).

O import de hy3dgen.texgen (que puxa o custom_rasterizer CUDA) é PREGUIÇOSO:
o app abre normalmente mesmo se o rasterizer ainda não tiver sido compilado;
o erro só aparece (claro) quando o usuário pede textura.
"""

import os
import sys
import gc
import time
import traceback

import numpy as np
import torch
import trimesh

# Auto-suficiente: garante os modelos locais e o pacote hy3dgen importável mesmo
# se este módulo for usado/testado fora do app.py (que também configura isso).
_BASE = os.path.dirname(os.path.abspath(__file__))
_HUNYUAN_DIR = os.path.join(_BASE, "Hunyuan3D2")
if os.path.isdir(_HUNYUAN_DIR) and _HUNYUAN_DIR not in sys.path:
    sys.path.insert(0, _HUNYUAN_DIR)
os.environ.setdefault("HY3DGEN_MODELS", os.path.join(_BASE, "models", "hy3d"))
os.environ.setdefault("HF_HOME", os.path.join(_BASE, "models"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(_BASE, "models"))

# pipeline residente (carregar os modelos de difusão é caro: cache por sessão)
_paint_pipe = None
_paint_key = None  # (subfolder, render_size, texture_size)


def _free_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def decimate_mesh(mesh, target_faces):
    """Decima via pymeshlab (robusto e já instalado). O trimesh 4.x exigiria
    'fast_simplification' (ausente aqui)."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        ), "m")
        # nome novo (pymeshlab >= 2022.2) com fallback para o antigo
        try:
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=int(target_faces),
                preservenormal=True, preservetopology=True, autoclean=True)
        except Exception:
            ms.simplification_quadric_edge_collapse_decimation(
                targetfacenum=int(target_faces), preservenormal=True)
        m = ms.current_mesh()
        return trimesh.Trimesh(
            vertices=m.vertex_matrix(), faces=m.face_matrix(), process=False)
    except Exception:
        print("[WARN] decimação falhou; uso a malha cheia:\n" + traceback.format_exc())
        return mesh


def load_paint_pipeline(subfolder="hunyuan3d-paint-v2-0-turbo",
                        render_size=1024, texture_size=1024, low_vram=True):
    """Carrega (ou reaproveita) o Hunyuan3DPaintPipeline. Se só a resolução
    mudou, troca apenas o renderer (barato) e mantém os modelos residentes."""
    global _paint_pipe, _paint_key
    key = (subfolder, int(render_size), int(texture_size))
    if _paint_pipe is not None and _paint_key == key:
        return _paint_pipe

    from hy3dgen.texgen import Hunyuan3DPaintPipeline
    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

    # só a resolução mudou -> recria o renderer e mantém os modelos
    if _paint_pipe is not None and _paint_key is not None and _paint_key[0] == subfolder:
        _paint_pipe.config.render_size = int(render_size)
        _paint_pipe.config.texture_size = int(texture_size)
        _paint_pipe.render = MeshRender(
            default_resolution=int(render_size), texture_size=int(texture_size))
        _paint_key = key
        return _paint_pipe

    _free_vram()
    print(f"[INFO] Carregando Hunyuan3D-Paint ({subfolder}) "
          f"render={render_size} tex={texture_size}...")
    pipe = Hunyuan3DPaintPipeline.from_pretrained(
        "tencent/Hunyuan3D-2", subfolder=subfolder)

    # sobrescreve a resolução padrão (config hardcoda 2048, pesado p/ 4 GB)
    pipe.config.render_size = int(render_size)
    pipe.config.texture_size = int(texture_size)
    pipe.render = MeshRender(
        default_resolution=int(render_size), texture_size=int(texture_size))

    if low_vram:
        # tira delight + multiview da VRAM quando ociosos (cabe em 4 GB)
        try:
            pipe.enable_model_cpu_offload()
            print("[INFO] paint: model_cpu_offload ativo")
        except Exception:
            print("[WARN] model_cpu_offload falhou (uso spill p/ RAM):\n"
                  + traceback.format_exc())
        # slicing reduz o pico de atenção/VAE — seguro, ajuda em 4 GB
        for k in ("delight_model", "multiview_model"):
            sub_pipe = getattr(pipe.models.get(k), "pipeline", None)
            if sub_pipe is None:
                continue
            try:
                sub_pipe.enable_attention_slicing()
            except Exception:
                pass
            vae = getattr(sub_pipe, "vae", None)
            if vae is not None:
                try:
                    vae.enable_slicing()  # API atual (substitui enable_vae_slicing)
                except Exception:
                    pass

    _paint_pipe = pipe
    _paint_key = key
    print("[INFO] Hunyuan3D-Paint pronto.")
    return pipe


def texture_mesh(mesh, image_rgba,
                 subfolder="hunyuan3d-paint-v2-0-turbo",
                 render_size=1024, texture_size=1024,
                 max_faces=120000, progress=None):
    """Repinta `mesh` a partir de `image_rgba` (PIL RGBA, fundo removido).
    Retorna um trimesh com TextureVisuals (UV + imagem de textura)."""
    def _p(v, d):
        if progress is not None:
            progress(v, desc=d)

    _p(0.05, "Preparando malha (decimação para baking)...")
    work = mesh.copy()
    n0 = len(work.faces)
    work = decimate_mesh(work, max_faces)
    print(f"[INFO] paint: faces {n0} -> {len(work.faces)} (alvo {max_faces})")

    _p(0.18, f"Carregando Hunyuan3D-Paint ({subfolder})...")
    pipe = load_paint_pipeline(subfolder, render_size, texture_size)

    _p(0.4, "Hunyuan-Paint: delight + multiview + baking (lento em 4 GB)...")
    if image_rgba.mode != "RGBA":
        image_rgba = image_rgba.convert("RGBA")
    with torch.no_grad():
        textured = pipe(work, image=image_rgba)

    _free_vram()
    _p(0.97, "Textura assada.")
    return textured


def texture_mesh_from_files(mesh_path, rgba_path,
                            subfolder="hunyuan3d-paint-v2-0-turbo",
                            render_size=1024, texture_size=1024,
                            max_faces=120000, progress=None):
    """Igual a texture_mesh, mas lê o mesh (.ply) e a imagem (.png) do disco.
    Usado no lote (fase 2): o paint carrega 1× (cache _paint_pipe) e é
    reaproveitado para todos os itens. .ply preserva o frame e as cores e
    volta como Trimesh puro (sem Scene)."""
    from PIL import Image
    mesh = trimesh.load(mesh_path, process=False, force='mesh')
    rgba = Image.open(rgba_path).convert("RGBA")
    return texture_mesh(mesh, rgba, subfolder=subfolder,
                        render_size=render_size, texture_size=texture_size,
                        max_faces=max_faces, progress=progress)


def get_texture_image(textured_mesh):
    """Extrai a imagem (PIL) da textura de um trimesh texturizado."""
    vis = getattr(textured_mesh, "visual", None)
    if vis is None:
        return None
    mat = getattr(vis, "material", None)
    if mat is not None:
        for attr in ("image", "baseColorTexture"):
            img = getattr(mat, attr, None)
            if img is not None:
                return img
    return getattr(vis, "image", None)


def export_result(textured_mesh, base_mesh, mode, fmt, out_dir):
    """Exporta conforme o modo escolhido. Retorna (caminho, nome_arquivo).
      mode='model'    -> só geometria (usa base_mesh; funciona sem textura)
      mode='texture'  -> só a imagem da textura (.png)
      mode='both'     -> modelo + textura (GLB embute tudo; OBJ vira .zip)
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = time.time_ns()
    fmt = (fmt or "glb").lower().lstrip(".")
    if fmt not in ("glb", "obj", "ply"):
        fmt = "glb"

    if mode == "texture":
        img = get_texture_image(textured_mesh)
        if img is None:
            raise RuntimeError("Não há textura para exportar — gere a textura primeiro.")
        path = os.path.join(out_dir, f"texture_{ts}.png")
        img.save(path)
        return path, os.path.basename(path)

    if mode == "model":
        src = base_mesh if base_mesh is not None else textured_mesh
        geom = trimesh.Trimesh(vertices=src.vertices, faces=src.faces, process=False)
        path = os.path.join(out_dir, f"model_{ts}.{fmt}")
        geom.export(path)
        return path, os.path.basename(path)

    # mode == 'both'
    if textured_mesh is None:
        raise RuntimeError("Não há modelo texturizado — gere a textura primeiro.")
    if fmt == "obj":
        import zipfile
        import shutil
        d = os.path.join(out_dir, f"textured_{ts}")
        os.makedirs(d, exist_ok=True)
        textured_mesh.export(os.path.join(d, "model.obj"))  # obj + mtl + png
        zpath = os.path.join(out_dir, f"textured_{ts}.zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in os.listdir(d):
                z.write(os.path.join(d, f), f)
        shutil.rmtree(d, ignore_errors=True)
        return zpath, os.path.basename(zpath)

    # GLB (ou ply) — GLB embute a textura num único arquivo
    ext = "glb" if fmt == "obj" else fmt
    path = os.path.join(out_dir, f"textured_{ts}.{ext}")
    textured_mesh.export(path)
    return path, os.path.basename(path)
