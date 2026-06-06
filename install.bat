@echo off
setlocal enabledelayedexpansion

echo =================================================
echo  3D Generator PRO - Instalador (TripoSR wrapper)
echo =================================================
echo.

set "ENV_NAME=env"
set "PROJECT_DIR=%~dp0"
set "ENV_DIR=%PROJECT_DIR%%ENV_NAME%"

:: -------- Python --------
echo [1/6] Verificando Python...
where python >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado no PATH. Instale Python 3.10 ou 3.11 ^(recomendado^)
    echo e tente novamente. Python 3.13 pode ter problemas com algumas dependencias.
    pause
    exit /b 1
)
python --version
echo.

:: -------- venv --------
echo [2/6] Criando ambiente virtual "%ENV_NAME%"...
if not exist "%ENV_DIR%\Scripts\activate.bat" (
    python -m venv "%ENV_DIR%"
) else (
    echo (ja existe, reutilizando)
)
call "%ENV_DIR%\Scripts\activate.bat"

:: -------- pip --------
echo.
echo [3/6] Atualizando pip / setuptools / wheel...
python -m pip install --upgrade pip setuptools wheel

:: -------- PyTorch --------
echo.
echo [4/7] Instalando PyTorch CUDA 12.4 (fallback CPU)...
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo (fallback para CPU)
    python -m pip install torch torchvision torchaudio
)

:: -------- Dependencias do app --------
echo.
echo [5/7] Instalando dependencias do app...
python -m pip install ^
    "omegaconf>=2.3.0" ^
    "einops>=0.7.0" ^
    "transformers>=4.35.0" ^
    "diffusers>=0.27" ^
    "accelerate" ^
    "trimesh>=4.0.5" ^
    "rembg" ^
    "huggingface-hub" ^
    "gradio>=4.0" ^
    "imageio[ffmpeg]" ^
    "pillow" ^
    "scikit-image" ^
    "opencv-python" ^
    "pymeshlab" ^
    "pygltflib" ^
    "xatlas" ^
    "numpy" ^
    "onnxruntime"

:: onnxruntime CPU (a versao -gpu exige cuDNN 9 e quebra o rembg)
python -m pip uninstall -y onnxruntime-gpu 1>nul 2>nul

:: -------- TripoSR --------
echo.
echo [6/7] Verificando codigo do TripoSR...
where git >nul 2>&1
if errorlevel 1 (
    echo [ERRO] git nao encontrado. Instale o Git ou copie as pastas manualmente.
    pause
    exit /b 1
)
if not exist "%PROJECT_DIR%TripoSR\tsr\system.py" (
    echo Clonando TripoSR...
    git clone https://github.com/VAST-AI-Research/TripoSR.git "%PROJECT_DIR%TripoSR"
) else (
    echo (codigo TripoSR ja presente)
)

:: -------- Hunyuan3D-2 (alta qualidade) --------
echo.
echo [7/7] Verificando codigo do Hunyuan3D-2...
if not exist "%PROJECT_DIR%Hunyuan3D2\hy3dgen\shapegen\pipelines.py" (
    echo Clonando Hunyuan3D-2...
    git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git "%PROJECT_DIR%Hunyuan3D2"
) else (
    echo (codigo Hunyuan3D2 ja presente)
)

:: Pre-baixa o modelo Hunyuan-2mini-turbo (~3.8GB) para o diretorio local
:: que o app espera (evita travamento de lock/download no primeiro uso).
echo Baixando modelo Hunyuan3D-2mini-turbo (uma vez, ~3.8GB)...
set "HY3DGEN_MODELS=%PROJECT_DIR%models\hy3d"
python -c "import os;from huggingface_hub import snapshot_download;d=os.path.join(r'%PROJECT_DIR%models','hy3d','tencent','Hunyuan3D-2mini');os.makedirs(d,exist_ok=True);snapshot_download(repo_id='tencent/Hunyuan3D-2mini',allow_patterns=['hunyuan3d-dit-v2-mini-turbo/*'],local_dir=d,max_workers=4)"
if errorlevel 1 (
    echo [AVISO] Download do Hunyuan falhou. Sera baixado no primeiro uso do modo Alta Qualidade.
) else (
    echo [OK] Modelo Hunyuan3D pronto.
)

:: -------- torchmcubes opcional --------
echo.
echo Tentando instalar torchmcubes ^(opcional - acelera marching cubes em GPU^)...
set FORCE_CUDA=0
python -m pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git 2>nul
if errorlevel 1 (
    echo [AVISO] torchmcubes nao foi instalado ^(requer compilador C++^).
    echo         O sistema usara scikit-image como fallback automatico.
) else (
    echo [OK] torchmcubes instalado.
)

echo.
echo =================================================
echo  Instalacao concluida.
echo  Execute "run.bat" para iniciar o aplicativo.
echo =================================================
pause
endlocal
