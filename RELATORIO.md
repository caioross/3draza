# RELATÓRIO ULTRADETALHADO — [SAAS] 3draza

> Auditoria técnica completa do sistema "3D Generator PRO" (wrapper Gradio em torno do TripoSR).
> Data: 2026-05-15 (v1-v2), 2026-05-16 (v3 — habilitação GPU)
> Versão: 3 (pós-ativação CUDA na RTX 3050)

---

## 0. APÊNDICE v3 — ATIVAÇÃO DA GPU (2026-05-16)

Após as correções v2, o sistema estava logicamente correto mas **rodava em CPU**
(PyTorch instalado era `2.11.0+cpu`). Hardware real detectado:

- **GPU:** NVIDIA GeForce RTX 3050 Laptop — **4 GB VRAM**
- **Visual Studio 2022 Community + C++ tools** presente (compilação possível)
- **CUDA Toolkit v13.1 (nvcc)** presente
- Python 3.13.2 (e 3.12 disponível); sem 3.10/3.11

### Bugs P0 adicionais encontrados e corrigidos na v3

| # | Problema | Correção |
|---|----------|----------|
| G1 | PyTorch CPU-only — GPU nunca usada | Reinstalado `torch 2.6.0+cu124` (uninstall + index cu124; o `--upgrade` simples NÃO troca a build) |
| G2 | `onnxruntime-gpu` exige cuDNN 9 ausente → `Cannot load symbol cudnnCreate` ao remover fundo | Trocado por `onnxruntime` (CPU) + `rembg.new_session("u2net", providers=["CPUExecutionProvider"])` |
| G3 | PyTorch não carrega seu próprio cuDNN no Windows: `Could not locate cudnn_graph64_9.dll` no forward GPU | `_enable_torch_cudnn_dlls()` em `app.py` põe `torch/lib` no **PATH** antes de `import torch` (cuDNN 9 carrega plugins via LoadLibrary, ignora add_dll_directory) |
| G4 | `UnicodeEncodeError` (emoji ✅) no console Windows cp1252 | `sys.stdout/stderr.reconfigure(encoding="utf-8")` |
| G5 | **Caminho do projeto contém `[SAAS]`** — `glob.glob` interpretava como classe de caracteres e não achava `torch/lib` | Removido glob; checagem direta com `os.path.isdir` |
| G6 | rembg rebaixava `u2net.onnx` ignorando `models/u2net.onnx` | `os.environ["U2NET_HOME"] = models/` |

### Resultado medido (pipeline real via `app.py`, imagem `robot.png`)

```
DEVICE = cuda  |  load 9.6s  |  geração 5.5s
mesh: 80.637 vértices / 161.270 faces  |  GLB 3.2 MB
VRAM pico: 2.49 GB  (cabe nos 4 GB com folga)  |  Zero erros
```

Ganho: de ~60-120 s (CPU) para **~5 s (GPU)**. Sistema TripoSR-GPU estável.

---

## 0b. APÊNDICE v4 — DOIS MOTORES + SOTA PESADO (2026-05-16)

Atendendo ao pedido de "state-of-the-art" e aceitando o trade-off de 4 GB,
foi integrado um segundo motor de alta qualidade.

### Arquitetura final (dois motores, com fallback)

| Motor | Modelo | Tempo | Detalhe | VRAM | Uso |
|-------|--------|-------|---------|------|-----|
| **Rápido** | TripoSR | ~5 s | ~80k verts | 2.5 GB | Default, confiável |
| **Alta qualidade** | Hunyuan3D-2mini-turbo (0.6B, step-distilled) | ~155 s | **~237k verts / ~862k faces** | 4.69 GB (usa RAM compartilhada) | SOTA shape |

`app.generate_3d` roteia pelo motor escolhido na UI. Se o Hunyuan falhar por
qualquer motivo, **fallback automático para TripoSR** — o usuário nunca fica
sem resultado (requisito "sem erros").

### Bugs P0 da v4 encontrados e corrigidos

| # | Problema | Correção |
|---|----------|----------|
| H1 | `enable_model_cpu_offload()` do Hunyuan referencia `self.components` inexistente → crash | Abandonado offload (frágil); usa config de VRAM fixa |
| H2 | `enable_model_cpu_offload` + `__call__` manual → tensores em cpu e cuda | idem H1 — sem offload |
| H3 | **Hang infinito** em `from_pretrained`: `smart_load_model` faz `snapshot_download` sem `cache_dir`/resume; processos mortos deixavam lock órfão do huggingface_hub | `HY3DGEN_MODELS` aponta p/ dir local fixo + download resumível único em `load_hunyuan()` → carga **offline**, nunca trava |
| H4 | Pico VRAM 4.69 GB > 4.29 GB físicos | Aceito: Windows faz spill p/ RAM compartilhada; parâmetros (`octree_resolution=256`, `num_chunks=3000`) mantêm o pico estável independente da imagem; lento mas **sem erro** |
| H5 | Diagnóstico mascarado: `| tail -N` bufferiza até EOF (parecia "travado") | Lição de método, não do código |

### Resultado final medido (Hunyuan via `app.generate_3d`, `robot.png`)

```
total 155.1s | 237.407 vértices / 861.742 faces | GLB 13 MB
VRAM pico 4.69 GB | carga OFFLINE do modelo local | zero erros
```

### Por que NÃO TRELLIS / Hunyuan3D-2 full

- 4 GB VRAM: TRELLIS (~16 GB) e Hunyuan3D-2 full (~10-24 GB) não cabem nem
  com spill — OOM duro. mini-turbo (0.6B) é o maior SOTA que roda aqui.
- Texturização (texgen) precisa de rasterizador CUDA custom compilado;
  mantido **shape-only** (geometria), que é estável.

### Bug de runtime R1 (descoberto ao rodar `run.bat` de verdade)

**Sintoma:** `ModuleNotFoundError: No module named 'onnxruntime'` →
`SystemExit: 1` no `import rembg`, mesmo com onnxruntime instalado no venv.

**Causa raiz:** o venv `env\` foi criado originalmente em `E:\3draza\env`
e depois movido para `E:\Projetos\[SAAS] 3draza\env`. Os scripts
`Scripts\activate.bat` de um venv têm o caminho **hardcoded**; ao mover a
pasta, a ativação falha silenciosamente e `python` cai no **Python global**
(`C:\Python313` + `%APPDATA%\Python`), que tem torch/gradio mas **não**
onnxruntime → rembg aborta com `sys.exit(1)`.

**Correção:** `run.bat` agora chama `env\Scripts\python.exe` por **caminho
absoluto** (o `python.exe` do venv funciona mesmo movido; só os `activate`
quebram) e define `PYTHONNOUSERSITE=1` para bloquear o site-packages do
usuário. Validado: `python = env\Scripts\python.exe`, `rembg` e
`onnxruntime 1.26.0` corretos, geração TripoSR OK.

## 0c. APÊNDICE v5 — PRESETS DE QUALIDADE (2026-05-16)

Usuário priorizou **qualidade** (tempo não é problema). Mediu-se o impacto
das alavancas (imagem `robot.png`):

| Modelo | passos | octree | tempo gen | vértices | VRAM |
|--------|--------|--------|-----------|----------|------|
| turbo | 5 | 256 | ~57 s | 237k | 4.69 GB |
| turbo | 5 | 384 | ~57 s | 538k | 4.91 GB |
| turbo | 5 | 512 | ~92 s | 966k | 5.85 GB |
| **mini (não-turbo)** | 30 | 512 | ~92 s | 982k | 5.98 GB |
| **mini (não-turbo)** | 50 | 512 | ~166 s | 1.04M | 6.09 GB |

**Conclusões:**
- `octree_resolution` é a alavanca de densidade de malha (256→512 ≈ 4× vértices).
- O modelo **não-turbo** (não-distilado) a 30-50 passos melhora a **fidelidade
  da forma** à imagem (não o nº de vértices) — é o maior ganho perceptual.
- VRAM 6 GB num cartão de 4 GB: funciona via spill p/ RAM compartilhada
  (Windows WDDM); lento mas **sem erro**, como o usuário aceitou.
- Carga do modelo não-turbo (`model.fp16.ckpt`, 3.8 GB, `use_safetensors=False`)
  ~95 s, uma vez por sessão.

**Presets implementados na UI** (`HQ_PRESETS`):

| Preset | Modelo | passos | octree | ~tempo |
|--------|--------|--------|--------|--------|
| Equilíbrio | turbo | 5 | 256 | ~1 min |
| Alta | turbo | 5 | 384 | ~1-2 min |
| **Máxima** (default) | mini | 30 | 512 | ~4-6 min |
| Extrema | mini | 50 | 512 | ~8-10 min |

+ slider de `octree_resolution` (0-768) nas opções avançadas para override
manual (com ajuste automático de `num_chunks` para conter o pico de VRAM).
Só uma variante do modelo fica residente por vez (cabe melhor em 4 GB);
trocar de preset recarrega a variante necessária.

## 0d. APÊNDICE v6 — DETALHE EXTREMO / ANTI-QUADRICULADO (2026-05-16)

Feedback: dedos/nariz/cabelo mal feitos e **"quadriculado" ao dar zoom**.
São dois problemas distintos:

### Problema 1 — "quadriculado" (staircase do marching cubes)
A malha segue a grade de voxels do Marching Cubes → degraus visíveis no zoom.

- **Tentativa SOTA (DMC / `diso.DiffDMC`):** extrator de superfície dual que
  posiciona vértices por gradiente (superfície lisa). **FALHOU ao compilar:**
  CUDA do sistema é **13.1**, PyTorch foi compilado com **12.4** → mismatch
  (`The detected CUDA version (13.1) mismatches ... (12.4)`). Exigiria
  instalar o toolkit CUDA 12.4 (invasivo). Documentado como escalonamento.
- **Solução aplicada (sem compilação, robusta):** pós-processamento
  `_refine_mesh` com **suavização Taubin** (`trimesh.smoothing.filter_taubin`,
  λ=0.5, μ=-0.53). Taubin é passa-banda: remove o staircase de alta
  frequência **sem encolher o volume** nem borrar características reais.
  Precedido de limpeza (`nondegenerate_faces`, `unique_faces`,
  `remove_unreferenced_vertices` — API trimesh 4.x). Níveis na UI:
  Desligado / Suave (8 it) / **Forte (18 it, default)** / Máximo (35 it).
  Medido (Máxima+Forte): 910k verts, 1.82M faces, 230 s, 5.85 GB, sem erro.

### Problema 2 — detalhe fino (dedos/nariz/cabelo)
Limite de capacidade do modelo. Testou-se o **Hunyuan3D-2 full (1.1B)**
(`tencent/Hunyuan3D-2`, `hunyuan3d-dit-v2-0`, fp16 = 4.93 GB).
**Resultado: INVIÁVEL em 4 GB.** O safetensors fp16 (4.93 GB) sozinho já
excede a VRAM total (4.29 GB) — só carregar força spill massivo;
GPU travou em 100 % util / 3873 MiB por 10+ min sem terminar nem o load.
Rodaria, mas a ~30-60 min/geração com risco real de **WDDM TDR**
(reset do driver) — impraticável. Modelo removido (liberou 5 GB).

**Teto prático de qualidade neste hardware:** `hunyuan3d-dit-v2-mini`
(não-distilado, 0.6B) + octree 512 + suavização Taubin Forte =
~910k-1M vértices, superfície lisa, ~4 min, estável. É o melhor
custo/qualidade que cabe em 4 GB. Para o full 1.1B / TRELLIS seria
necessária uma GPU de ≥ 12 GB.

### Limpeza de disco
`models/` tinha 31 GB (downloads `.cache` duplicados do HF em modo
`local_dir`, variantes fp32 baixadas à toa, cópias HF-cache obsoletas
de quando o Hunyuan ainda não usava `HY3DGEN_MODELS`). Reduzido para
**14 GB** removendo apenas lixo seguro (scratch `.cache`, fp32 não
usados, dupes). TripoSR + Hunyuan (turbo e não-turbo) revalidados OK
pós-limpeza.

### Estado dos requisitos do usuário

- [x] Subir imagem → objeto 3D: **funciona** (2 motores)
- [x] "Não pode ter erros": fallback automático + carga offline (sem hang)
- [x] "Ver o progresso corretamente": `gr.Progress` em etapas nomeadas
- [x] "Funcionar perfeitamente": validado end-to-end nos 2 motores
- [x] "Mais inovador/moderno": Hunyuan3D-2mini-turbo (SOTA viável em 4 GB)
- [x] "Não importa se demorar": HQ ~155 s, aceitável e estável

---

## 0e. APÊNDICE v7 — TEXTURIZAÇÃO (Hunyuan3D-Paint) (2026-06-03)

Pedido: após gerar o 3D, **opcionalmente gerar a textura a partir da imagem**, na
melhor qualidade possível, e poder exportar **só o modelo / só a textura /
modelo + textura**.

### Solução: Hunyuan3D-Paint (`hy3dgen.texgen`)
Pega o mesh gerado (TripoSR ou Hunyuan) + a imagem e repinta o modelo com
**difusão multi-view**: delight (remove sombra/luz) → multiview UNet (6 vistas) →
back-projection + baking numa textura UV (xatlas). Gera detalhe plausível até
nos lados não vistos. Funciona para qualquer mesh; alinhamento é nativo para os
meshes do Hunyuan (frame calibrado pela Tencent).

### Bugs/blockers P0 encontrados e resolvidos

| # | Problema | Correção |
|---|----------|----------|
| T1 | `MeshRender` hardcoda `raster_mode='cr'` → `import custom_rasterizer` (extensão CUDA) sem fallback PyTorch | Compilado o `custom_rasterizer` |
| T2 | nvcc do sistema é **13.1**, PyTorch é **cu124** → mismatch de major version bloqueia `CUDAExtension` (mesmo erro do `diso`/v6) | Instalado **CUDA Toolkit 12.4** ao lado do 13.1 |
| T3 | winget recusa instalar 12.4 (vê `Nvidia.CUDA` 13.1 "já instalado"); shell sem admin → instalador aborta | `INSTALAR_CUDA_12.4.bat` auto-elevável (UAC) usando o instalador já no cache do winget |
| T4 | Caminho pip do CUDA (`nvidia-*-cu12`) **inviável no Windows**: wheel só traz `ptxas.exe` (sem `nvcc.exe`) e nenhum `.lib` | Abandonado; usado o toolkit real |
| T5 | Path do projeto tem espaço (`SAAS 3draza`) → quebra as command lines do nvcc/cl | Build numa pasta temporária sem espaços; `.pyd` instalado no venv |
| T6 | MSVC 14.41 (1941) rejeitado pelo `host_config.h` do CUDA 12.4 | Forçado toolset **14.38** (`_MSC_VER` 1938) via `vcvars64 -vcvars_ver=14.38` |
| T7 | `import custom_rasterizer_kernel` falha com "DLL load failed" | Precisa `import torch` antes (registra `torch/lib`); no app/texture_gen já ocorre |
| T8 | Em 4 GB, os construtores põem delight+multiview na VRAM antes do offload | `enable_model_cpu_offload` + slicing + **liberar TripoSR/Hunyuan antes de texturizar** + spill WDDM |
| T9 | Loader custom do unet (`modules.py`) carrega **só `diffusion_pytorch_model.bin`**; ao limpar disco apaguei o `.bin` | `.bin` reconstruído filtrando o safetensors p/ as 1535 chaves do modelo (36 chaves ip-adapter extras descartadas; 0 faltando) |

### Otimizações para 4 GB
- `enable_model_cpu_offload()` nos sub-pipelines (delight + multiview).
- `enable_attention_slicing()` + `vae.enable_slicing()`.
- Decimação do mesh via **pymeshlab** antes do unwrap/baking (alvo configurável;
  `trimesh` exigiria `fast_simplification`, ausente).
- `_free_generator_models()` libera TripoSR/Hunyuan da VRAM antes do paint.
- Resolução de render/textura configurável (1024 padrão; 2048 = máxima).

### Arquivos
- **`texture_gen.py`** (novo): `load_paint_pipeline`, `texture_mesh`,
  `decimate_mesh`, `get_texture_image`, `export_result` (3 modos). Import de
  `hy3dgen.texgen` é preguiçoso (app abre mesmo sem rasterizer compilado).
- **`app.py`**: estado `LAST`, `generate_texture`, `export_download`,
  `_free_generator_models`, UI (botão 🎨 Gerar textura, resolução, nº de faces,
  seletor "O que baixar", preview 3D texturizado).
- **`install.bat`**: + `xatlas`.
- **`INSTALAR_CUDA_12.4.bat`** (raiz): instalador elevável do CUDA 12.4.

### Resultado medido (TripoSR, `robot.png`, 1024px, 80k faces)
```
mesh 80.637 v / 161.270 f -> decimado 80k
textura 1024x1024 (desvio 60.8 = conteúdo real) em 807 s
VRAM pico 10.72 GB (spill WDDM; 4 GB físicos) | 3 exports OK (glb/png/glb)
```

### Resultado medido (Hunyuan turbo "Equilíbrio", `robot.png`, 1024px, 100k faces)
```
mesh 242.678 v / 485.364 f (96 s) -> decimado p/ 100k
textura 1024x1024 (desvio 43.9 = conteúdo real) em 611 s
VRAM pico 9.03 GB (otimização _free_generator_models: -1.7 GB vs tripo)
3 exports OK: textured 3.2 MB (59k v, UV) / textura 0.8 MB / modelo 8.7 MB (242k v cheios)
```
Caminho principal (Hunyuan) validado end-to-end; alinhamento nativo correto.

### Modos de export
- **Modelo + textura** → GLB (textura embutida) ou OBJ+MTL+PNG (.zip).
- **Somente textura** → PNG do atlas UV.
- **Somente modelo** → geometria pura (glb/obj/ply).

### Como usar
1. Gerar o 3D normalmente (TripoSR ou Hunyuan).
2. Escolher resolução da textura (1024/2048) e nº de faces.
3. **🎨 Gerar textura** (lento em 4 GB — minutos, com spill).
4. Escolher "O que baixar" e **💾 Baixar**.

### Limitações neste hardware
- 4 GB: texturizar é lento (spill); 2048 ainda mais. Aceito (qualidade > tempo).
- Requer CUDA 12.4 + `custom_rasterizer` compilado (one-time).

---

## 0f. APÊNDICE v8 — LOTE + UI/UX + LOGGING/ETA (2026-06-03)

Pedido: processar **várias imagens (dezenas) uma a uma** (modelo 3D **e** textura); UI/UX melhor (o sistema ficou complexo); mais configs/ajustes/qualidade; logging melhor com **tempo decorrido (MM:SS) e ETA**. Regra: **não quebrar nada**.

### Princípio de zero-regressão
Tudo novo reaproveita os blocos já validados **sem alterar o comportamento**: `_gen_tripo`, `_gen_hunyuan`, `_remove_bg_rgba`, `texture_gen.texture_mesh`, `export_result`, `_free_generator_models`. O fluxo de imagem única continua idêntico, só reorganizado numa aba.

### Lote em 2 fases (eficiência em 4 GB)
- **Fase 1 (geometria):** modelo de forma carrega **1×**; gera o mesh de todas as imagens; salva cada um como **`.ply`** (frame-safe, preserva cor, `Trimesh` puro) + a RGBA sem fundo como `.png`. Solta a ref do mesh após cada item (RAM).
- `_free_generator_models()` entre fases.
- **Fase 2 (textura, opcional):** paint carrega **1×** (cache `_paint_pipe`) e texturiza/exporta cada `.ply`, um a um.
- **Tolerância a falha por imagem** (try/except): uma imagem ruim é pulada (log ❌) e o lote continua. Saída incremental em `outputs/batch_<ts>/NNN_<nome>/` + `manifest.json` + `log.txt`, zipados ao fim.

### Logging + tempo + ETA
- `run_batch` é um **gerador** que dá `yield (log, status, galeria, zip, viewer)` a cada etapa (fonte da verdade do log ao vivo).
- **Relógio ao vivo** via `gr.Timer(1s)` → `_tick_clock()` lê o dict global `PROGRESS` e mostra `⏱️ Decorrido MM:SS | ⏳ Restante ~MM:SS | etapa X/N`. Confirmado: o Timer dispara em paralelo ao gerador (eventos com `concurrency_id` distintos no Gradio 6.13). Timer escreve só no campo `clock`; o gerador escreve nos demais (sem corrida).
- **`_SubProgress`**: adaptador que mapeia o `progress(frac,desc)` dos `_gen_*` para uma sub-faixa da barra real — permite reusá-los sem editar.
- **ETA** = média seg/unidade concluída × restantes (unidade = 1 geometria + 1 textura por imagem).

### UI/UX (abas)
`build_ui` reescrito com `gr.Tabs` (componentes são globais ao Blocks → configs definidas 1× servem às duas abas):
- **🖼️ Imagem única** — fluxo atual.
- **📚 Lote** — upload múltiplo (`gr.File(file_count="multiple")`) + pasta opcional, toggle de textura, modo de export, log, galeria, `📦 baixar tudo (.zip)`, viewer.
- **⚙️ Configurações** (compartilhada) — motor, preset, **guidance_scale (novo)**, suavização, octree, MC res, remover fundo, foreground, formato, **resolução de textura 1024/1536/2048 (1536 novo)**, faces.

### Mais ajustes
- `guidance_scale` do Hunyuan exposto (era fixo 5.0) — append no fim de `_gen_hunyuan`/`generate_3d` (ordem posicional preservada, default 5.0 = idêntico).
- Resolução de textura agora 1024/1536/2048 (`_parse_tex_res`).

### Arquivos
- `app.py`: `PROGRESS`+helpers de tempo, `_tick_clock`, `_SubProgress`, `_collect_batch_inputs`, `_zip_dir`, `_write_manifest`, `run_batch`, `_parse_tex_res`, `guidance_scale`, `build_ui` em abas. Handlers de imagem única preservados.
- `texture_gen.py`: `texture_mesh_from_files` (lê `.ply`+`.png`, reusa `texture_mesh`).

### Validação
- Smoke `build_ui()` OK; contagens input↔param conferidas (generate_3d=10, run_batch=15in/5out).
- **Lote sem textura, 2 boas + 1 corrompida (TripoSR):** `✅ 2/3 modelos em 00:19`, corrompida pulada, zip+manifest+log OK, TripoSR 1×. Resiliência confirmada.
- **Imagem única (regressão):** `generate_3d` retorna 3 saídas, `LAST` setado — intacto.
- **Lote com textura (fase 2):** validado em seguida (paint 1×, export por modo, galeria).

---

## 1. VISÃO GERAL DO SISTEMA

### 1.1 O que o projeto pretende ser
Aplicativo desktop local (Windows) que:
1. Recebe uma imagem 2D pela interface Gradio.
2. Remove o fundo (rembg / U²-Net).
3. Usa o modelo **TripoSR** (Stability AI / VAST-AI) para reconstruir um mesh 3D
   single-image-to-3D (NeRF triplane + marching cubes).
4. Exporta um `.obj` para `outputs/` e exibe num visualizador 3D inline.

### 1.2 Estrutura de arquivos relevante
```
[SAAS] 3draza/
├── app.py                  ← UI Gradio (wrapper principal)
├── install.bat             ← Instalador
├── run.bat                 ← Inicializa o app
├── env/                    ← venv ativo (Python 3.13.2, torch CPU)
├── tripo_env/              ← venv vazio (criado pelo install.bat, NUNCA usado)
├── models/                 ← cache HF + u2net.onnx (rembg)
│   ├── u2net.onnx
│   └── xet/
├── outputs/                ← meshes gerados
├── temp/                   ← (vazio)
└── TripoSR/                ← repositório oficial clonado
    ├── tsr/
    │   ├── system.py       ← classe TSR
    │   ├── utils.py
    │   ├── bake_texture.py
    │   └── models/
    │       ├── isosurface.py        ← USA torchmcubes (NÃO INSTALADO)
    │       ├── nerf_renderer.py
    │       ├── network_utils.py
    │       ├── tokenizers/
    │       └── transformer/
    ├── requirements.txt
    ├── requirements_safe.txt
    ├── requirements_fixed.txt
    ├── run.py              ← CLI oficial de referência
    └── gradio_app.py       ← UI oficial de referência
```

---

## 2. BUGS CRÍTICOS (SISTEMA NÃO FUNCIONA)

### 2.1 ❌ `extract_mesh` chamado sem argumento obrigatório
**Arquivo:** `app.py:96`
```python
mesh = model.extract_mesh(scene_codes)[0]
```
**Problema:** A assinatura em `TripoSR/tsr/system.py:171` é
`extract_mesh(self, scene_codes, has_vertex_color, resolution=256, threshold=25.0)`.
`has_vertex_color` é **posicional obrigatório**. Toda chamada lança
`TypeError: extract_mesh() missing 1 required positional argument: 'has_vertex_color'`.

**Impacto:** O botão "Gerar 3D" sempre falha. Sistema 100% quebrado.

---

### 2.2 ❌ `import tsr` impossível — pacote não está no `sys.path`
**Arquivo:** `app.py:54, 85`
```python
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground
```
**Problema:** `app.py` está em `[SAAS] 3draza/` (raiz). O pacote `tsr/` vive
em `[SAAS] 3draza/TripoSR/tsr/`. Sem adicionar `TripoSR/` ao `sys.path`,
Python lança `ModuleNotFoundError: No module named 'tsr'`.

**Impacto:** O botão "Carregar Modelo" nunca passa do `import`.

---

### 2.3 ❌ `torchmcubes` não instalado (e impossível instalar nesse ambiente)
**Arquivo:** `TripoSR/tsr/models/isosurface.py:6`
```python
from torchmcubes import marching_cubes
```
**Diagnóstico real (verificado em runtime):**
```
ModuleNotFoundError: No module named 'torchmcubes'
torch 2.11.0+cpu   |   Python 3.13.2   |   CUDA disponível: False
```

**Por que não instala:**
- `install.bat` faz `pip install git+https://github.com/tatsy/torchmcubes.git` no FINAL,
  com `FORCE_CUDA=0`. Este pacote compila código C++/CUDA e exige
  MSVC Build Tools + matching PyTorch ABI.
- Python 3.13 não tem wheels pré-construídos e a build do source quase sempre
  falha em Windows sem toolchain instalado.
- O script não para nem alerta em caso de falha — install.bat termina como se
  estivesse OK.

**Impacto:** Mesmo se 2.1 e 2.2 forem resolvidos, `extract_mesh` falha no
import de `MarchingCubeHelper`.

---

### 2.4 ❌ Mismatch entre `install.bat` e `run.bat` (env errado)
- `install.bat` cria e instala em **`tripo_env\`**.
- `run.bat` ativa **`env\`**.

**Estado atual:** `tripo_env/` está vazio (sem pacotes); `env/` contém os pacotes
(provavelmente instalados manualmente depois). Mas qualquer usuário novo que
seguir o README óbvio (`install.bat` → `run.bat`) recebe `ModuleNotFoundError`
em tudo, porque o ambiente ativado nunca foi populado.

---

### 2.5 ❌ PyTorch instalado em **CPU**, install.bat pede **CUDA 11.8**
```
torch 2.11.0+cpu
```
**Causa raiz:** Python 3.13.2 — não há wheels CUDA 11.8 (`cu118`) para 3.13.
O pip silenciosamente resolve para a versão CPU.

**Impacto:** Inferência ~20× mais lenta. Mesmo com GPU física, roda em CPU.

---

### 2.6 ❌ Python 3.13 incompatível com pins do `install.bat`
`install.bat` pinou versões antigas (Pillow 10.0.1, transformers 4.35.0,
trimesh 4.0.5, omegaconf 2.3.0) feitas para Python 3.10/3.11. No 3.13:
- `transformers==4.35.0` falha install (incompatível com Py3.13).
- O `env/` que efetivamente funciona tem versões muito mais novas
  (transformers 5.7.0, trimesh 4.12.1, gradio 6.13.0, huggingface_hub 1.12.2)
  — instaladas à mão fora do script. Logo, o `install.bat` **não reproduz** o
  estado funcional do `env/`.

Além disso, os venvs apontam para `E:\3draza\...` (path antigo), enquanto o
projeto agora está em `E:\Projetos\[SAAS] 3draza\` — `pyvenv.cfg` aponta para
caminho correto, mas o `Scripts/activate.bat` interno pode ter o `VIRTUAL_ENV`
hardcoded para o path antigo, o que quebra `pip` e atalhos de console.

---

## 3. BUGS DE LÓGICA / IMPLEMENTAÇÃO INCOMPLETA

### 3.1 ⚠️ Pré-processamento incompleto vs. pipeline oficial
**Arquivo:** `app.py:75-109`
O `run.py` oficial (`TripoSR/run.py:140-147`) faz:
```python
image = remove_background(Image.open(image_path), rembg_session)
image = resize_foreground(image, args.foreground_ratio)
image = np.array(image).astype(np.float32) / 255.0
image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5  # composite cinza
image = Image.fromarray((image * 255.0).astype(np.uint8))
```
`app.py` **omite a composição cinza** (último passo). O modelo TripoSR foi
treinado com fundo cinza (0.5,0.5,0.5). Passar a RGBA crua produz reconstruções
ruins (artefatos, geometria distorcida).

### 3.2 ⚠️ `rembg_session` não é cacheado entre execuções
**Arquivo:** `app.py:88` → `remove_background(img)`
Sem `rembg_session=...`, cada geração reinicializa o ONNX Runtime + carrega o
U²-Net (~170 MB) do zero. Várias gerações seguidas = lentidão acumulada.

### 3.3 ⚠️ `model.renderer.set_chunk_size(...)` nunca chamado
Default em `nerf_renderer.py:33` é `chunk_size = 0` (sem chunking). Isso
provoca OOM em GPUs pequenas (<8 GB) ou consome muita RAM em CPU.

### 3.4 ⚠️ Sem `to_gradio_3d_orientation` no mesh exportado
O mesh é exportado direto sem aplicar a rotação que orienta o objeto
"de pé" no `gr.Model3D`. Resultado: visualização sempre tombada/deitada.

### 3.5 ⚠️ Não chama `model.renderer.set_chunk_size` antes de gerar
O `gradio_app.py` oficial faz isso com `chunk_size=8192`. Necessário aqui.

### 3.6 ⚠️ Tratamento de imagem RGB sem alpha
Se o usuário sobe um PNG sem alpha ou JPG, `remove_background` produz RGBA OK,
mas se `image.mode == "RGBA"` com alpha cheio, `do_remove = False` e o
`resize_foreground` (que assume 4 canais) **funciona** mas pode falhar se a
imagem é RGB sem alpha. Linha `assert image.shape[-1] == 4` em
`utils.py:422` quebra silenciosamente.

### 3.7 ⚠️ Variável global `model` sem lock
Em Gradio multi-usuário, dois cliques simultâneos em "Carregar Modelo"
geram race condition na inicialização. Para uso local single-user não é
problema, mas é fraqueza de design.

### 3.8 ⚠️ Sem `model.eval()`
Após `model.to(DEVICE)`, deveria-se chamar `model.eval()` para desligar
dropout/batchnorm em modo treino. O `system.py` herda `nn.Module` cujo
default é `training=True`. Mesmo se os submódulos não tiverem layers
sensíveis, é boa prática.

### 3.9 ⚠️ Não há retorno de erro estruturado / sem traceback
`except Exception as e: return None, f"Erro: {str(e)}"` esconde o stacktrace.
Debugging local impossível sem editar o código.

### 3.10 ⚠️ `MODEL_DIR` como `HUGGINGFACE_HUB_CACHE` mas modelo nunca foi pré-baixado
`models/` contém apenas `u2net.onnx` (rembg) — não o checkpoint TripoSR.
O primeiro `load_model()` baixa ~1.5 GB do HF. Sem rede = falha total.
Não há mensagem de "baixando…", o botão fica congelado.

### 3.11 ⚠️ Sem feedback de progresso
Gradio suporta `gr.Progress` — não usado. Geração leva 30–120 s; usuário
acha que travou.

### 3.12 ⚠️ Formato `.obj` sem material
Mesh é exportado como `.obj` sem MTL/cores. `gr.Model3D` mostra cinza puro.
O modelo gera cor por vértice — exportar como `.glb` preserva isso.

### 3.13 ⚠️ `outputs/` cresce sem limite
Cada `mesh_{timestamp}.obj` fica no disco. Não há rotação/cleanup nem
botão para baixar/remover. Após uso prolongado, GB de lixo.

### 3.14 ⚠️ `torch.cross(...)` sem `dim=` em utils.py
`utils.py:376-377`: `torch.cross(lookat, up)` sem `dim=-1`. Em torch ≥ 2.0
gera **UserWarning** e em alguma versão futura será erro.

### 3.15 ⚠️ Falta `model.eval()` + `torch.inference_mode()`
`generate_3d` usa `torch.no_grad()` (OK), mas `inference_mode()` é mais
eficiente em torch 2.x.

### 3.16 ⚠️ `gr.Image(type="pil")` retorna PIL.Image — OK, mas sem normalização EXIF
Fotos de celular vêm com rotação EXIF. Gradio passa a imagem rotacionada
errada. Falta `ImageOps.exif_transpose(img)`.

### 3.17 ⚠️ `torch.cuda.empty_cache()` chamado mesmo após falha
Linha 104: dentro do `try` — OK só roda no caminho feliz. Mas o `except`
não libera memória. Em loops de falha, vaza.

### 3.18 ⚠️ `gr.Model3D` recebe `.obj` mas Three.js (backend do Model3D) prefere `.glb`
Visualização inline tem performance ruim com `.obj` grande (sem normais
indexadas, sem compressão).

### 3.19 ⚠️ `install.bat` baixa `requirements_safe.txt` filtrando linhas, mas
**também precisa filtrar `moderngl` e `xatlas`** em sistemas sem GPU/drivers
OpenGL — moderngl exige driver moderno. Sem texture baking esses são
opcionais; mantê-los obriga instalação sem motivo.

### 3.20 ⚠️ `from tsr.system import TSR` dentro de `load_model` — adiada para tempo de execução
Padrão "lazy import" sem comentário. OK funcional, mas torna falhas tardias
em vez de no startup, dificultando diagnóstico.

### 3.21 ❌ `torch.load` sem `weights_only=False` (torch ≥ 2.6 default True)
**Arquivo:** `TripoSR/tsr/system.py:69`
```python
ckpt = torch.load(weight_path, map_location="cpu")
```
Desde torch 2.6, o **default de `weights_only` virou `True`**. O ambiente atual
roda torch 2.11.0. O checkpoint `model.ckpt` do TripoSR é um state_dict puro
(provavelmente OK), mas em alguns forks/versões inclui metadata serializada
com `pickle` que falha com weights_only=True. **Adicionar `weights_only=False`
explicitamente** elimina o risco e remove o warning ruidoso.

### 3.22 ⚠️ `HUGGINGFACE_HUB_CACHE` é legacy — usar `HF_HOME`
Em huggingface_hub ≥ 0.20 (atual: 1.12.2), a env var recomendada virou
`HF_HOME`. O sistema antigo ainda funciona, mas convém setar **ambas**
para garantir cache consistente entre versões.

### 3.23 ⚠️ ViT model é re-inicializado do zero a cada `from_pretrained`
**Arquivo:** `tsr/models/tokenizers/image.py:21-28`
`ViTModel(ViTModel.config_class.from_pretrained(...))` cria pesos aleatórios
do ViT. Os pesos reais vêm depois via `load_state_dict` (linha 70 de system.py).
Funciona, mas força download do `config.json` do DINO (`facebook/dino-vitb16`)
mesmo offline. Sem internet na primeira execução = falha. Mitigar pré-baixando
ou definindo `HF_HUB_OFFLINE=0`/cacheando.

### 3.24 ⚠️ `register_buffer(persistent=False)` para image_mean/std
OK na lógica, mas significa que esses tensores **não vêm no state_dict** —
são reinicializados em CPU. Após `model.to(DEVICE)` eles migram. Não é bug,
só observação de comportamento.

---

## 4. LACUNAS / FUNCIONALIDADES PELA METADE

| # | Item | Estado |
|---|------|--------|
| L1 | Botão "Carregar Modelo" requer clique manual; deveria ser autoload | Incompleto |
| L2 | Sem opção de escolher entre OBJ/GLB/PLY | Faltando |
| L3 | Sem controle de resolução do marching cubes (256 fixo) | Faltando |
| L4 | Sem controle do `foreground_ratio` (0.85 fixo) | Faltando |
| L5 | Sem botão "Remover fundo" separado (preview antes de gerar) | Faltando |
| L6 | Sem download direto do arquivo gerado | Faltando |
| L7 | Sem galeria/histórico de gerações | Faltando |
| L8 | Sem baking de textura (modelo gera só vertex colors) | Faltando |
| L9 | Sem fila/processamento batch de várias imagens | Faltando |
| L10 | Sem timer/log de duração mostrado na UI | Faltando |
| L11 | Sem checagem de VRAM/RAM antes de gerar | Faltando |
| L12 | Sem indicador visual de "modelo carregado" persistente | Faltando |
| L13 | Sem README.md / docs do projeto raiz | Faltando |
| L14 | `temp/` existe mas nunca é usado | Lixo / não implementado |
| L15 | `xet/` em `models/` é cache de huggingface — OK mas não documentado | Não documentado |
| L16 | Não há `requirements.txt` no root do projeto | Faltando |
| L17 | Não há `.gitignore` para excluir `env/`, `tripo_env/`, `outputs/` | Faltando |
| L18 | install.bat não detecta GPU NVIDIA / falha de CUDA | Faltando |
| L19 | Não há `--share`/`--port` opcional na UI | Faltando |
| L20 | Não há teste automatizado / smoke test | Faltando |

---

## 5. PROBLEMAS DE SEGURANÇA / ROBUSTEZ

- **S1** `app.launch(inbrowser=True)` sem `server_name`/`server_port` explícito → porta dinâmica imprevisível; OK em local.
- **S2** Nenhum limite de tamanho de upload — usuário sobe imagem 8K → OOM.
- **S3** Nome do arquivo gerado usa `int(time.time())` (segundos). Dois cliques no mesmo segundo sobrescrevem. Usar `time.time_ns()` ou UUID.
- **S4** Em multi-user share, `outputs/` é exposto inteiro pelo Gradio.

---

## 6. PROBLEMAS DE PERFORMANCE

- **P1** Sem `set_chunk_size`. (cf. 3.3)
- **P2** Sem `torch.compile` (torch 2.11 suporta) — ganho 1.3–1.8× em inferência.
- **P3** rembg recarrega U²-Net a cada chamada. (cf. 3.2)
- **P4** Marching cubes default 256³ = 16 M vértices avaliados. Sem chunking
  consome ~5 GB RAM. Em CPU isso é ~25-60 s.
- **P5** Resultado em CPU é viável mas lento. Sem GPU detectada o usuário
  não tem opção de baixar resolução para 128.

---

## 7. ESTADO DA ARTE (referência)

O fluxo state-of-the-art para single-image-to-3D em 2026 inclui:

| Etapa | Implementação ideal |
|-------|---------------------|
| Background removal | `rembg` com `isnet-general-use` (≥ U²-Net), sessão reaproveitada |
| Pré-processamento | exif_transpose, resize foreground, composite cinza |
| Modelo | TripoSR (OK, é o atual SOTA open-source single-view fast) |
| Marching cubes | torchmcubes (GPU) **ou** `pymcubes` (CPU, pip puro, sem build) |
| Pós-processamento | mesh simplification (decimate), normais suaves, rotação Gradio |
| Texturização | atlas baking via xatlas + moderngl (vértices → textura UV) |
| Export | `.glb` (com vertex colors), `.obj` opcional |
| UI | Gradio com `gr.Progress`, autoload, controles avançados num accordion |

O projeto atual cobre as **etapas centrais** mas tem os bugs listados acima.

---

## 8. PLANO DE CORREÇÃO (ordem de execução)

### Fase A — Restaurar funcionamento (P0)
1. **`isosurface.py`**: adaptador de marching cubes (torchmcubes → pymcubes → skimage), garantindo zero compilação C++.
2. **`app.py`**: 
   - inserir `TripoSR/` em `sys.path`;
   - passar `has_vertex_color=True` em `extract_mesh`;
   - aplicar composição cinza (igual `run.py`);
   - cachear `rembg_session`;
   - `set_chunk_size(8192)`;
   - `model.eval()` + `torch.inference_mode()`;
   - exportar `.glb` com vertex colors;
   - aplicar `to_gradio_3d_orientation`;
   - `exif_transpose` na entrada;
   - autoload em segundo plano;
   - `gr.Progress` com etapas;
   - traceback completo no status em caso de erro;
   - nomes de arquivo com `time.time_ns()`.
3. **Instalar `pymcubes`** no `env\` ativo.
4. **`run.bat`**: trocar `env\Scripts\activate` por `tripo_env\Scripts\activate`? Não — o env funcional é o `env\`. **Manter `env\`** e corrigir `install.bat` para também usar `env\` (consistência).

### Fase B — UX / qualidade (P1)
5. Controles: resolução MC (slider 64–384), foreground_ratio, formato de export.
6. Botão de remover fundo separado (preview).
7. Galeria de outputs com download.
8. Indicador persistente "Modelo: carregado/não carregado".

### Fase C — Polimento (P2)
9. `README.md` no root.
10. `.gitignore`.
11. Cleanup automático de `outputs/` >24h.
12. Smoke test.

---

## 9. VERIFICAÇÃO DE PREMISSAS

Antes de corrigir, validei:
- ✅ Python no `env\` é 3.13.2 e funcional (pyvenv.cfg + run de teste).
- ✅ torch 2.11.0+cpu importa.
- ❌ torchmcubes NÃO importa (confirmado em runtime).
- ✅ rembg, omegaconf, trimesh, gradio, transformers, huggingface_hub presentes.
- ❌ checkpoint TripoSR não está em `models/` (será baixado no primeiro load).
- ✅ `models/u2net.onnx` presente (rembg roda offline se HOME estiver configurado).
- ⚠️ `tripo_env/` vazio — instalador documentado nunca produziu estado funcional.

---

## 10. RISCOS RESIDUAIS

- **R1** Primeiro `load_model` baixa ~1.5 GB do HuggingFace. Sem rede falha.
  Mitigação: cachear em `models/` (já configurado via env var) e instruir
  usuário a rodar uma vez online.
- **R2** Sem GPU, geração leva 30–120 s. Acima do timeout default do Gradio
  (que não tem timeout, mas o navegador pode considerar perdido). Mitigação:
  `gr.Progress` mostra progresso ativo.
- **R3** `pymcubes` (substituto do `torchmcubes`) é ~30% mais lento mas
  funciona em pip puro. Aceito como trade-off por confiabilidade.
- **R4** Vertex colors via GLB exigem trimesh ≥ 3.20 — presente (4.12.1).

---

## 11. CHECKLIST DE ACEITAÇÃO

Após correções, o sistema deve:
- [x] `run.bat` ativa o venv certo (`env\`) e abre o navegador.
- [x] Modelo TripoSR é carregado sob demanda (na primeira geração ou via botão).
- [x] Status mostra device + MC backend + estado de carregamento.
- [x] Upload de JPG/PNG/PNG-com-alpha funciona (com `ImageOps.exif_transpose`).
- [x] Fundo é removido corretamente; **botão de preview** opcional disponível.
- [x] Marching cubes funciona via fallback automático (**scikit-image**) sem precisar compilar nada.
- [x] Mesh exportado em `.glb` (default) **com cores por vértice**; OBJ e PLY também disponíveis.
- [x] Visualizador 3D aplica `to_gradio_3d_orientation` (objeto em pé).
- [x] `outputs/` recebe arquivo nomeado com `time.time_ns()` (sem colisão).
- [x] Erros mostram **traceback completo** no status.
- [x] Controles UI: remover fundo on/off, foreground ratio, resolução MC, formato.
- [x] `gr.Progress` indica progresso em 4 etapas (preprocess / scene codes / mesh / export).
- [x] Sessão rembg cacheada entre execuções; modelo cacheado em variável global.

---

## 12. RESUMO DAS CORREÇÕES APLICADAS

| # | Arquivo | Correção |
|---|---------|----------|
| 1 | `TripoSR/tsr/models/isosurface.py` | Backend MC com fallback: torchmcubes → pymcubes → scikit-image |
| 2 | `TripoSR/tsr/system.py` | `torch.load(..., weights_only=False)` |
| 3 | `TripoSR/tsr/utils.py` | `torch.cross(..., dim=-1)` para silenciar deprecation |
| 4 | `app.py` | Reescrita completa — ver lista abaixo |
| 5 | `run.bat` | Ativa `env\` (não `tripo_env\`), valida existência, paths absolutos |
| 6 | `install.bat` | venv consistente (`env\`), CUDA 12.1 + fallback CPU, deps modernas, torchmcubes opcional, scikit-image como fallback garantido |

**Mudanças no `app.py`:**
- Adiciona `TripoSR/` ao `sys.path` antes de qualquer import de `tsr`.
- Seta `HF_HOME` e `HUGGINGFACE_HUB_CACHE` apontando para `models/`.
- `load_model()` faz `set_chunk_size(8192)` + `model.eval()`.
- `generate_3d()` recebe controles: `do_remove_bg`, `foreground_ratio`, `mc_resolution`, `export_format`.
- Pré-processamento alinhado ao `run.py` oficial (EXIF, remove bg, resize, composite cinza 0.5).
- `extract_mesh(scene_codes, has_vertex_color=True, resolution=…)` — assinatura correta.
- Sessão rembg cacheada em variável global.
- `torch.inference_mode()` para máxima eficiência.
- `to_gradio_3d_orientation(mesh)` antes do export.
- Export GLB/OBJ/PLY selecionável.
- `time.time_ns()` no nome.
- `gr.Progress` em quatro etapas.
- Exceções convertidas em traceback no status.
- Botão extra "Preview (remover fundo)" e botão "Recarregar modelo".
- `app.queue().launch(inbrowser=True, server_name="127.0.0.1")` (porta local explícita).

---

## 13. VERIFICAÇÃO PÓS-CORREÇÃO

Smoke test executado no `env\`:
```
DEVICE: cpu
tsr imports OK
MC backend: skimage
UI build OK
```
Marching cubes sintético (esfera em volume 64³) gerou
6 744 vértices / 13 484 faces, range [0.20, 0.80] — consistente com esfera de raio 0.3 centrada em 0.5.

**Bloqueios remanescentes (dependem de rede / hardware):**
- Primeiro `load_model` baixa ~1.5 GB de `stabilityai/TripoSR` no HuggingFace
  (esperado, cacheado em `models/`).
- Sem GPU NVIDIA, geração de mesh em 256³ leva ~30-90 s em CPU.
  Reduzir `mc_resolution` para 128 acelera ~4×.
- `torchmcubes` continua disponível como opcional via `install.bat` para
  quem tiver MSVC + CUDA, mas o fallback `scikit-image` garante funcionamento.

---

*Fim do relatório (v2 — pós-correção).*
