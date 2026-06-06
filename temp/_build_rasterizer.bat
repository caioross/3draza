@echo off
setlocal
:: =====================================================================
:: Compila custom_rasterizer (extensao CUDA do Hunyuan3D-Paint).
:: CUDA 12.4 (casa com torch cu124) + MSVC 14.38 (_MSC_VER 1938, aceito
:: pelo host_config.h do 12.4). Compila numa pasta SEM espacos porque o
:: caminho do projeto ("SAAS 3draza") quebra as command lines do nvcc/cl.
:: =====================================================================

set "PROJ=%~dp0.."
set "SRC=%PROJ%\Hunyuan3D2\hy3dgen\texgen\custom_rasterizer"
set "PY=%PROJ%\env\Scripts\python.exe"
set "BUILD=%TEMP%\cr_build"

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
set "CUDA_HOME=%CUDA_PATH%"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"
set "TORCH_CUDA_ARCH_LIST=8.6"
set "DISTUTILS_USE_SDK=1"
set "PYTHONNOUSERSITE=1"

if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo [ERRO] nvcc 12.4 ausente em "%CUDA_PATH%".
    exit /b 2
)

:: copia o codigo para pasta sem espacos
rmdir /s /q "%BUILD%" 2>nul
xcopy /e /i /q /y "%SRC%" "%BUILD%" >nul
if errorlevel 1 ( echo [ERRO] copia para "%BUILD%" falhou. & exit /b 3 )

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
if errorlevel 1 ( echo [ERRO] vcvars64.bat falhou. & exit /b 1 )

echo ===== nvcc em uso =====
nvcc --version | findstr /i release

echo ===== compilando custom_rasterizer (em %BUILD%) =====
cd /d "%BUILD%"
"%PY%" -m pip install . --no-build-isolation --no-deps --force-reinstall

echo ===== verificando import =====
"%PY%" -c "import custom_rasterizer, custom_rasterizer_kernel; print('IMPORT_OK custom_rasterizer')"
exit /b %errorlevel%
