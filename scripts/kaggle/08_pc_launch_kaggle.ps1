# =====================================================================
# Push le notebook sur Kaggle + lance le run automatiquement (CLI).
#
# Usage (depuis PowerShell, n'importe ou) :
#   .\scripts\kaggle\08_pc_launch_kaggle.ps1
#
# Pre-requis :
#   - Kaggle CLI installe (cf setup_kaggle.ps1)
#   - kaggle.json OU access_token configure
#   - Le secret GITHUB_TOKEN doit etre attache au notebook UNE FOIS via l'UI
#     (https://www.kaggle.com/code/abdelhalimnssiri/mrgrt-train-unet-fold0
#      -> Add-ons -> Secrets -> coche GITHUB_TOKEN)
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Push Kaggle notebook + lancement du run" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Localiser le kaggle CLI ----------------------------------------
$kaggleBin = $null
$cmd = Get-Command kaggle -ErrorAction SilentlyContinue
if ($cmd) {
    $kaggleBin = $cmd.Source
} else {
    foreach ($v in @("Python313", "Python312", "Python311", "Python310")) {
        $candidate = "$env:USERPROFILE\AppData\Roaming\Python\$v\Scripts\kaggle.exe"
        if (Test-Path $candidate) { $kaggleBin = $candidate; break }
    }
}
if (-not $kaggleBin) {
    Write-Host "[ERREUR] kaggle CLI introuvable." -ForegroundColor Red
    Write-Host "  Lance d'abord : .\scripts\kaggle\setup_kaggle.ps1"
    exit 1
}
Write-Host "kaggle CLI : $kaggleBin"

# --- Dossier du notebook -------------------------------------------
$kaggleDir = $PSScriptRoot
$notebook  = Join-Path $kaggleDir "train_kaggle.ipynb"
$metadata  = Join-Path $kaggleDir "kernel-metadata.json"

if (-not (Test-Path $notebook)) {
    Write-Host "[ERREUR] $notebook absent." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $metadata)) {
    Write-Host "[ERREUR] $metadata absent." -ForegroundColor Red
    exit 1
}

# --- Verifier les variables critiques dans le notebook -------------
$content = Get-Content $notebook -Raw
if ($content -match '<TON_USER_GITHUB>') {
    Write-Host "[ERREUR] Le notebook contient encore <TON_USER_GITHUB>." -ForegroundColor Red
    Write-Host "  Edite manuellement : GITHUB_USER = `"NSSIRI`""
    exit 1
}
if ($content -notmatch 'mrgrt-oar-thorax-clean') {
    Write-Host "[ATTENTION] Le notebook ne pointe pas sur mrgrt-oar-thorax-clean." -ForegroundColor Yellow
    Write-Host "  Verifie KAGGLE_INPUT_DATASET dans la cellule de config."
}
Write-Host "Notebook   : OK (variables a jour)"
Write-Host ""

# --- Lire l'ID depuis le metadata ----------------------------------
$meta = Get-Content $metadata -Raw | ConvertFrom-Json
$kernelId = $meta.id
$kernelUrl = "https://www.kaggle.com/code/$kernelId"
Write-Host "Notebook   : $kernelId"
Write-Host "URL        : $kernelUrl"
Write-Host ""

# --- Push (cree le notebook OU le met a jour ET lance le run) ------
Write-Host "[etape 1/2] Push du notebook (kaggle kernels push)..." -ForegroundColor Yellow
Write-Host "  Note : si c'est la 1ere fois, le run echouera car GITHUB_TOKEN"
Write-Host "  n'est pas encore attache. Ne pas paniquer : on l'attachera"
Write-Host "  via l'UI une fois apres le 1er push, puis on re-push."
Write-Host ""

Push-Location $kaggleDir
try {
    & $kaggleBin kernels push -p .
    $pushExit = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($pushExit -ne 0) {
    Write-Host ""
    Write-Host "[ERREUR] Push a echoue." -ForegroundColor Red
    Write-Host "  Causes possibles :"
    Write-Host "   - Authentification : verifier ~/.kaggle/kaggle.json"
    Write-Host "   - Dataset 'mrgrt-oar-thorax-clean' pas encore Available"
    Write-Host "   - Quota Kaggle GPU depasse (30h/semaine)"
    exit 1
}

Write-Host ""
Write-Host "[etape 2/2] Push reussi. Le run est lance automatiquement." -ForegroundColor Green
Write-Host ""

# --- Status check + lien -------------------------------------------
Write-Host "Statut du dernier run :"
& $kaggleBin kernels status $kernelId
Write-Host ""

Write-Host "============================================" -ForegroundColor Green
Write-Host " RUN LANCE." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Suivre l'avancement :"
Write-Host "  $kernelUrl"
Write-Host ""
Write-Host "Verifier le statut depuis PowerShell :"
Write-Host "  & '$kaggleBin' kernels status $kernelId"
Write-Host ""
Write-Host "Telecharger les outputs une fois fini :"
Write-Host "  & '$kaggleBin' kernels output $kernelId -p .\kaggle_output"
Write-Host ""

# --- Premier run : rappel sur le secret ----------------------------
Write-Host "IMPORTANT - 1er push :" -ForegroundColor Yellow
Write-Host "  Si tu n'as JAMAIS attache le secret GITHUB_TOKEN a ce notebook,"
Write-Host "  le run va echouer a l'etape [0]. Pour fixer :"
Write-Host "  1) Ouvre $kernelUrl"
Write-Host "  2) Edit -> Add-ons -> Secrets -> coche GITHUB_TOKEN"
Write-Host "  3) Relance ce script (.\scripts\kaggle\08_pc_launch_kaggle.ps1)"
Write-Host ""
