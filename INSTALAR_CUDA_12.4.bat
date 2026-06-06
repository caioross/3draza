@echo off
title Instalar CUDA Toolkit 12.4 (necessario para a textura 3D)

:: ---- auto-elevar: se nao for admin, relanca com UAC ----
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Solicitando permissao de administrador ^(UAC^)...
    echo Clique em SIM na janela que vai aparecer.
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ============================================================
echo  Instalando CUDA Toolkit 12.4 (silencioso).
echo  Pode levar de 3 a 8 minutos. NAO feche esta janela
echo  ate aparecer a mensagem CONCLUIDO.
echo ============================================================
echo.

set "EXE=C:\Users\Matrix\AppData\Local\Temp\WinGet\Nvidia.CUDA.12.4\cuda_12.4.1_551.78_windows.exe"
set "DONE=E:\Projetos\SAAS 3draza\temp\cuda_install_done.txt"
del "%DONE%" >nul 2>&1

if exist "%EXE%" (
    echo Usando o instalador ja baixado no cache do winget...
    "%EXE%" -s
) else (
    echo Cache nao encontrado; baixando e instalando via winget...
    winget install --id Nvidia.CUDA --version 12.4 --force --accept-package-agreements --accept-source-agreements --silent
)

echo.
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
    echo OK> "%DONE%"
    echo ============================================================
    echo  [CONCLUIDO] CUDA 12.4 instalado com sucesso!
    echo  Pode fechar esta janela e voltar ao Claude.
    echo ============================================================
) else (
    echo FAIL> "%DONE%"
    echo ============================================================
    echo  [FALHOU] o nvcc 12.4 nao foi encontrado apos a instalacao.
    echo  Volte ao Claude e me avise para tentarmos outro caminho.
    echo ============================================================
)
echo.
pause
