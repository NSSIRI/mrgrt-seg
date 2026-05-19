# =====================================================================
# Telecharge l'archive du dataset clean (303 patients) depuis MARWAN
# vers le PC, puis l'extrait.
#
# Pre-requis :
#   - Avoir d'abord lance sur MARWAN :
#       bash scripts/transfer/06_marwan_create_archive.sh
#     (cree /scratch/users/a.nssiri/mrgrt-seg/downloads/data_thorax_complet.tar.gz)
#   - Avoir un raccourci SSH "marwan" configure dans ~/.ssh/config (sinon
#     remplacer "marwan" par "a.nssiri@hpc-login.marwan.ma" dans ce script).
#
# Usage (depuis PowerShell, dans le dossier du projet) :
#   .\scripts\transfer\07_pc_download_clean.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Download dataset clean (303 patients)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# --- Config ---
$marwanHost     = "marwan"   # ou "a.nssiri@hpc-login.marwan.ma" si pas de config SSH
$remoteArchive  = "/scratch/users/a.nssiri/mrgrt-seg/downloads/data_thorax_complet.tar.gz"
$projectRoot    = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$localArchive   = Join-Path $projectRoot "data_thorax_complet.tar.gz"
$localExtract   = Join-Path $projectRoot "data_thorax_complet"

Write-Host "Source   : ${marwanHost}:$remoteArchive"
Write-Host "PC dest  : $localArchive"
Write-Host ""

# --- 1. scp -----------------------------------------------------------
if (Test-Path $localArchive) {
    $existSize = (Get-Item $localArchive).Length / 1MB
    Write-Host "Archive deja presente : $localArchive ($([math]::Round($existSize,1)) MB)" -ForegroundColor Yellow
    $ans = Read-Host "Re-telecharger ? (O/N)"
    if ($ans -eq "O" -or $ans -eq "o") {
        Remove-Item $localArchive -Force
    }
}

if (-not (Test-Path $localArchive)) {
    Write-Host ""
    Write-Host "[etape 1/2] Telechargement (scp)..." -ForegroundColor Yellow
    Write-Host "  Cela prend 5-30 min selon ta vitesse internet."
    Write-Host ""

    scp "${marwanHost}:${remoteArchive}" "$localArchive"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] scp a echoue." -ForegroundColor Red
        Write-Host "  Verifie : "
        Write-Host "  - Le fichier existe sur MARWAN (ls $remoteArchive)"
        Write-Host "  - Tu peux faire 'ssh $marwanHost' sans erreur"
        Write-Host "  - Si pas de config SSH 'marwan', remplace la ligne marwanHost en haut du script"
        exit 1
    }

    $sizeNew = (Get-Item $localArchive).Length / 1MB
    Write-Host "  Telecharge : $([math]::Round($sizeNew,1)) MB"
}

# --- 2. Extract -------------------------------------------------------
Write-Host ""
Write-Host "[etape 2/2] Extraction..." -ForegroundColor Yellow

if (Test-Path $localExtract) {
    Write-Host "Dossier existant : $localExtract" -ForegroundColor Yellow
    $ans = Read-Host "Le supprimer avant extraction ? (O/N)"
    if ($ans -eq "O" -or $ans -eq "o") {
        Remove-Item $localExtract -Recurse -Force
    } else {
        Write-Host "Extraction annulee. Archive conservee : $localArchive"
        exit 0
    }
}

# tar est inclus dans Windows 10+ par defaut (bsdtar)
tar -xf "$localArchive" -C $projectRoot
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERREUR] Extraction echouee." -ForegroundColor Red
    Write-Host "  Tu peux extraire manuellement avec 7-Zip ou via Explorer (clic droit -> Extraire tout)."
    exit 1
}

# --- 3. Copier le metadata Kaggle pour le clean dataset ---------------
$metaSrc = Join-Path $projectRoot "scripts\kaggle\dataset_metadata_clean_303.json"
$metaDst = Join-Path $localExtract "dataset-metadata.json"
if (Test-Path $metaSrc) {
    Copy-Item $metaSrc $metaDst
    Write-Host "  dataset-metadata.json copie dans $localExtract"
}

# --- Bilan ------------------------------------------------------------
$nPatients = (Get-ChildItem $localExtract -Directory -Filter "PATIENT_*" | Measure-Object).Count
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " TERMINE." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host " Dossier  : $localExtract"
Write-Host " Patients : $nPatients"
Write-Host ""
Write-Host "Pour uploader sur Kaggle :"
Write-Host "  kaggle datasets create -p `"$localExtract`" --dir-mode zip"
Write-Host ""
Write-Host "Tu peux maintenant supprimer l'archive (gain ~1 GB) :"
Write-Host "  Remove-Item `"$localArchive`""
