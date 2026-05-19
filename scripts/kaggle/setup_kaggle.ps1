# =====================================================================
# Setup Kaggle CLI + upload du dataset baseline (250 patients)
# Usage (depuis PowerShell, dans le dossier du projet) :
#   .\scripts\kaggle\setup_kaggle.ps1
#
# Si l'execution PowerShell est bloquee par defaut, lancer une seule fois :
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
# (repondre Y a la confirmation)
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Setup Kaggle CLI + upload baseline (n=250)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- 1. Verifier que Python est installe -----------------------------
try {
    $pyVer = python --version 2>&1
    Write-Host "[OK] Python detecte : $pyVer"
} catch {
    Write-Host "[ERREUR] Python n'est pas dans le PATH." -ForegroundColor Red
    Write-Host "  Installe Python 3.10+ depuis https://www.python.org/downloads/"
    Write-Host "  Pendant l'install, coche 'Add Python to PATH'."
    exit 1
}

# --- 2. Installer / mettre a jour le CLI Kaggle ----------------------
Write-Host ""
Write-Host "[etape 1/5] Installation du CLI Kaggle..." -ForegroundColor Yellow
pip install --upgrade --user kaggle | Out-Host
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERREUR] pip install a echoue." -ForegroundColor Red
    exit 1
}

# --- 3. Creer le dossier .kaggle et placer kaggle.json ---------------
Write-Host ""
Write-Host "[etape 2/5] Placement de kaggle.json..." -ForegroundColor Yellow

$kaggleDir = Join-Path $env:USERPROFILE ".kaggle"
$kaggleJson = Join-Path $kaggleDir "kaggle.json"
$downloads = Join-Path $env:USERPROFILE "Downloads\kaggle.json"

if (-not (Test-Path $kaggleDir)) {
    New-Item -ItemType Directory -Path $kaggleDir | Out-Null
    Write-Host "  Cree : $kaggleDir"
}

if (Test-Path $kaggleJson) {
    Write-Host "  kaggle.json deja present dans $kaggleDir. On ne touche pas."
} elseif (Test-Path $downloads) {
    Move-Item -Path $downloads -Destination $kaggleJson
    Write-Host "  Deplace depuis Downloads vers $kaggleDir"
} else {
    Write-Host "[ERREUR] kaggle.json introuvable." -ForegroundColor Red
    Write-Host "  Va sur https://www.kaggle.com/settings -> API -> Create New Token"
    Write-Host "  Le fichier doit etre dans ton dossier Downloads ou deja dans $kaggleDir"
    exit 1
}

# --- 4. Tester l'authentification ------------------------------------
Write-Host ""
Write-Host "[etape 3/5] Test de l'authentification Kaggle..." -ForegroundColor Yellow

$kaggleBin = "$env:USERPROFILE\AppData\Roaming\Python\Python313\Scripts\kaggle.exe"
if (-not (Test-Path $kaggleBin)) {
    # Essayer les versions Python communes
    foreach ($v in @("Python312", "Python311", "Python310", "Python39")) {
        $candidate = "$env:USERPROFILE\AppData\Roaming\Python\$v\Scripts\kaggle.exe"
        if (Test-Path $candidate) { $kaggleBin = $candidate; break }
    }
}
if (-not (Test-Path $kaggleBin)) {
    # Fallback : utiliser "kaggle" du PATH si install systeme
    $kaggleBin = "kaggle"
}

try {
    & $kaggleBin --version | Out-Host
} catch {
    Write-Host "[ERREUR] kaggle CLI non trouve dans le PATH." -ForegroundColor Red
    Write-Host "  Ajoute manuellement %USERPROFILE%\AppData\Roaming\Python\Python3XX\Scripts au PATH"
    exit 1
}

# --- 5. Demander le username Kaggle et mettre a jour metadata --------
Write-Host ""
Write-Host "[etape 4/5] Configuration du metadata du dataset..." -ForegroundColor Yellow

$kaggleUser = Read-Host "Entre ton username Kaggle (visible dans l'URL de ton profil)"
if ([string]::IsNullOrWhiteSpace($kaggleUser)) {
    Write-Host "[ERREUR] Username vide." -ForegroundColor Red
    exit 1
}
$kaggleUser = $kaggleUser.Trim().ToLower() -replace "[^a-z0-9_-]",""
Write-Host "  Username sanitise : $kaggleUser"

# Mise a jour metadata baseline (250 patients)
$metaBaseline = Join-Path $PSScriptRoot "..\..\data\dataset-metadata.json"
$metaBaseline = (Resolve-Path $metaBaseline).Path
$content = Get-Content $metaBaseline -Raw
$content = $content -replace "TON_USER_KAGGLE", $kaggleUser
Set-Content -Path $metaBaseline -Value $content -NoNewline
Write-Host "  Mis a jour : $metaBaseline"

# Mise a jour metadata clean 303 (pour plus tard)
$metaCleanSource = Join-Path $PSScriptRoot "dataset_metadata_clean_303.json"
if (Test-Path $metaCleanSource) {
    $contentClean = Get-Content $metaCleanSource -Raw
    $contentClean = $contentClean -replace "TON_USER_KAGGLE", $kaggleUser
    Set-Content -Path $metaCleanSource -Value $contentClean -NoNewline
    Write-Host "  Mis a jour : $metaCleanSource"
}

# --- 6. Upload du dataset baseline ------------------------------------
Write-Host ""
Write-Host "[etape 5/5] Upload du dataset baseline (250 patients, ~1.9 GB)..." -ForegroundColor Yellow
Write-Host "  Cela prend 10-25 min selon ton upload internet."
Write-Host ""

$dataDir = Join-Path $PSScriptRoot "..\..\data"
$dataDir = (Resolve-Path $dataDir).Path
Write-Host "  Source : $dataDir"

$confirm = Read-Host "Lancer l'upload maintenant ? (O/N)"
if ($confirm -ne "O" -and $confirm -ne "o" -and $confirm -ne "Y" -and $confirm -ne "y") {
    Write-Host "Annule. Pour lancer manuellement plus tard :"
    Write-Host "  $kaggleBin datasets create -p $dataDir --dir-mode zip"
    exit 0
}

& $kaggleBin datasets create -p $dataDir --dir-mode zip
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERREUR] L'upload a echoue." -ForegroundColor Red
    Write-Host "  Causes possibles :"
    Write-Host "   - kaggle.json invalide -> retelecharge depuis Kaggle Settings"
    Write-Host "   - username Kaggle incorrect dans le metadata"
    Write-Host "   - dataset deja existant -> utiliser 'kaggle datasets version' a la place"
    exit 1
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " UPLOAD TERMINE." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Verifie sur :"
Write-Host "  https://www.kaggle.com/datasets/$kaggleUser/mrgrt-oar-thorax-baseline"
Write-Host ""
Write-Host "Le statut passe de 'Processing' a 'Available' apres ~5-15 min."
Write-Host "Une fois Available, on pourra l'attacher au notebook Kaggle."
