"""Valida o custom_rasterizer rodando na GPU (kernel CUDA), sem os modelos de
difusão: MeshRender + xatlas num cubo simples."""
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import texture_gen  # configura sys.path (Hunyuan3D2) + env; importa torch
import torch
import trimesh
from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender
from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

print("torch cuda:", torch.cuda.is_available(), flush=True)
try:
    r = MeshRender(default_resolution=512, texture_size=512)
    m = trimesh.creation.box(extents=(1, 1, 1))
    m = mesh_uv_wrap(m)
    r.load_mesh(m)
    nrm = r.render_normal(0, 0, return_type='pl')
    pos = r.render_position(0, 0, return_type='pl')
    print(f"render_normal={nrm.size} render_position={pos.size}", flush=True)
    print("RASTER_GPU_OK", flush=True)
except Exception:
    print("RASTER_FAIL:\n" + traceback.format_exc(), flush=True)
    sys.exit(1)
