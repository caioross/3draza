@echo off
setlocal

set "PY=%~dp0env\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERRO] Python do ambiente nao encontrado em "%PY%".
    echo Execute install.bat primeiro.
    pause
    exit /b 1
)

:: Usa o python.exe do venv por caminho absoluto. NAO depende do activate.bat,
:: que tem o caminho antigo hardcoded se o venv foi movido de pasta.
:: PYTHONNOUSERSITE bloqueia pacotes do site do usuario (%APPDATA%\Python)
:: que poluiriam o ambiente (ex.: rembg sem onnxruntime).
set "PYTHONNOUSERSITE=1"

"%PY%" "%~dp0app.py"

pause
endlocal
