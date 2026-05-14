@echo off
chcp 65001 >nul
title Setup local MRgRT - venv + PyTorch CPU + smoke test
color 0B
echo.
echo ============================================================
echo   Setup local MRgRT-Seg
echo   (venv + PyTorch CPU + MONAI + smoke test)
echo ============================================================
echo.
echo Repertoire courant : %CD%
echo.

REM --- Verifier Python ---
where python >nul 2>&1
if errorlevel 1 (
    color 0C
    echo ERREUR : Python non trouve dans le PATH.
    echo Redemarrez ce script dans un NOUVEAU terminal apres install Python.
    pause
    exit /b 1
)
echo --- Version Python ---
python --version
echo.

REM --- Creation/reparation venv ---
REM Si .venv existe mais activate.bat est absent, on supprime et recree.
if exist .venv (
    if not exist .venv\Scripts\activate.bat (
        echo --- .venv casse detecte, suppression et recreation ---
        rmdir /s /q .venv
    ) else (
        echo --- .venv existe et est OK, on le reutilise ---
    )
)
if not exist .venv (
    echo --- Creation environnement virtuel .venv ---
    python -m venv .venv
    if errorlevel 1 (
        color 0C
        echo ERREUR : impossible de creer .venv
        echo Verifier que Python est bien installe et accessible :
        python -m venv --help
        pause
        exit /b 1
    )
)

REM --- Verification finale que activate.bat est bien la ---
if not exist .venv\Scripts\activate.bat (
    color 0C
    echo ERREUR : .venv\Scripts\activate.bat introuvable apres creation.
    echo Le module venv de Python est peut-etre absent. Essayer :
    echo   python -m pip install --user virtualenv
    echo   python -m virtualenv .venv
    pause
    exit /b 1
)
echo.

REM --- Activation venv ---
call .venv\Scripts\activate.bat
if errorlevel 1 (
    color 0C
    echo ERREUR : impossible d'activer .venv
    pause
    exit /b 1
)
echo --- venv active ---
where python
echo.

REM --- Mise a jour pip ---
echo --- Mise a jour pip ---
python -m pip install --upgrade pip --quiet
echo.

REM --- Install PyTorch CPU ---
echo ============================================================
echo  1/3 - PyTorch + torchvision (version CPU)
echo  Telechargement ~250 Mo, peut prendre quelques minutes
echo ============================================================
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    color 0C
    echo ERREUR : install PyTorch echouee. Verifier connexion internet.
    pause
    exit /b 1
)
echo.

REM --- Install autres deps ---
echo ============================================================
echo  2/3 - MONAI, nibabel, scipy, sklearn, jupyter, etc.
echo ============================================================
pip install -r requirements_cpu.txt
if errorlevel 1 (
    color 0C
    echo ERREUR : install requirements echouee.
    pause
    exit /b 1
)
echo.

REM --- Smoke test ---
echo ============================================================
echo  3/3 - Smoke test
echo ============================================================
python smoke_test.py
set EXITCODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXITCODE%==0 (
    color 0A
    echo  SUCCESS : votre env local est pret !
    echo ============================================================
    echo.
    echo Pour reutiliser le venv plus tard :
    echo   cd %CD%
    echo   .venv\Scripts\activate
    echo.
    echo Pour lancer le notebook :
    echo   jupyter notebook notebooks\00_pedagogical_pipeline.ipynb
    echo.
) else (
    color 0C
    echo  ECHEC du smoke test - voir messages ci-dessus.
    echo ============================================================
)
pause
