# =====================================================================
# Script tout-en-un pour evaluer les 5 folds des 2 modeles :
#   1) Cree 2 batch eval notebooks (1 par compte) via Kaggle CLI
#   2) Polle leur execution
#   3) Telecharge les 10 CSV (5 folds x 2 modeles)
#   4) Agrege en un tableau final + tests Wilcoxon par OAR
#
# Pre-requis :
#   - Kaggle CLI installe (Python311\Scripts\kaggle.exe)
#   - kaggle_compte1.json et kaggle_compte2.json dans ~/.kaggle/
#   - scripts/kaggle/batch_eval.py committe et push sur GitHub
#   - Tous les notebooks training mrgrt-train-{model}-foldN existent (deja fait)
#
# Usage (PowerShell, dans le dossier du projet) :
#   .\scripts\kaggle\10_run_batch_eval_and_aggregate.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Lenovo\Desktop\mrgrt_seg"
$Kaggle = "C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts\kaggle.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Set-Location $ProjectRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " BATCH EVAL 5 folds x 2 modeles" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Backup kaggle.json actuel (compte 1 par defaut) -----------------
$kaggleConf = "$env:USERPROFILE\.kaggle\access_token"
$kaggleConf1 = "$env:USERPROFILE\.kaggle\access_token_compte1"
$kaggleConf2 = "$env:USERPROFILE\.kaggle\access_token_compte2"

if (-not (Test-Path $kaggleConf1)) {
    Copy-Item $kaggleConf $kaggleConf1
    Write-Host "Backup compte 1 OK"
}

if (-not (Test-Path $kaggleConf2)) {
    Write-Host "[ATTENTION] $kaggleConf2 ABSENT" -ForegroundColor Yellow
    Write-Host "  Tu dois sauvegarder le access_token du compte 2 :"
    Write-Host "  1. Login compte 2 (nssiri02) sur Kaggle"
    Write-Host "  2. https://www.kaggle.com/settings -> API -> Create New Token"
    Write-Host "  3. Sauve le token via :"
    Write-Host "     [System.IO.File]::WriteAllText('$kaggleConf2', 'KGAT_xxxx', [System.Text.ASCIIEncoding]::new())"
    Write-Host ""
    $ans = Read-Host "Continuer avec compte 1 seulement (UNet) ? (O/N)"
    if ($ans -ne "O" -and $ans -ne "o") { exit 0 }
    $SkipCompte2 = $true
} else {
    $SkipCompte2 = $false
}

# --- Helper : push un kernel via CLI --------------------------------
function Invoke-KaggleBatchEval {
    param(
        [string]$Username,        # abdelhalimnssiri ou nssiri02
        [string]$Model,           # unet ou segresnet
        [string]$TokenSourcePath  # path vers access_token a utiliser (compte 1 ou 2)
    )

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host " $($Model.ToUpper()) batch eval ($Username)" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow

    # Switch token (compte 1 ou 2)
    Copy-Item $TokenSourcePath $kaggleConf -Force

    $kernelId = "$Username/mrgrt-eval-$Model-batch"
    $tempDir = Join-Path $env:TEMP "kaggle_batch_eval_$Model"
    if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    # Generer batch_eval.ipynb + kernel-metadata.json via helper Python
    # (UTF-8 sans BOM, JSON garanti correct)
    $makeNb = Join-Path $ProjectRoot "scripts\kaggle\make_batch_notebook.py"
    if (-not (Test-Path $makeNb)) {
        Write-Host "[ERREUR] $makeNb absent" -ForegroundColor Red
        return $false
    }

    python $makeNb --model $Model --username $Username --out_dir $tempDir 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] generation notebook a echoue" -ForegroundColor Red
        return $false
    }

    # Push
    Write-Host "Push du kernel..."
    Push-Location $tempDir
    try {
        & $Kaggle kernels push -p . 2>&1 | Out-Host
    } finally {
        Pop-Location
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] Push echec" -ForegroundColor Red
        return $false
    }

    # Poll status
    Write-Host "Polling toutes les 60s (max 60 min)..."
    $start = Get-Date
    while ($true) {
        Start-Sleep -Seconds 60
        $elapsed = ((Get-Date) - $start).TotalMinutes
        $st = & $Kaggle kernels status $kernelId 2>&1
        Write-Host ("  [{0:F1} min] {1}" -f $elapsed, $st)
        if ($st -match "COMPLETE") { break }
        if ($st -match "ERROR|CANCELLED|FAILED") {
            Write-Host "[ERREUR] Run echec : $st" -ForegroundColor Red
            return $false
        }
        if ($elapsed -gt 60) {
            Write-Host "[TIMEOUT] 60 min depasses" -ForegroundColor Yellow
            return $false
        }
    }

    # Download
    Write-Host "Telechargement outputs..."
    $outDir = Join-Path $ResultsDir "batch_${Model}"
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    New-Item -ItemType Directory -Path $outDir | Out-Null
    & $Kaggle kernels output $kernelId -p $outDir 2>&1 | Out-Null
    Write-Host "Outputs dans : $outDir"

    return $true
}

# --- Run UNet (compte 1) -----------------------------------------------
$unetOK = Invoke-KaggleBatchEval -Username "abdelhalimnssiri" -Model "unet" -TokenSourcePath $kaggleConf1

# --- Run SegResNet (compte 2) -----------------------------------------
$segOK = $false
if (-not $SkipCompte2) {
    $segOK = Invoke-KaggleBatchEval -Username "nssiri02" -Model "segresnet" -TokenSourcePath $kaggleConf2
}

# --- Restore compte 1 -------------------------------------------------
Copy-Item $kaggleConf1 $kaggleConf -Force
Write-Host ""
Write-Host "kaggle.json restaure (compte 1)"

# --- Bilan -----------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " BATCH EVAL TERMINE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "UNet      : $(if ($unetOK) {'OK'} else {'ECHEC'})"
Write-Host "SegResNet : $(if ($segOK) {'OK'} elseif ($SkipCompte2) {'SKIP'} else {'ECHEC'})"
Write-Host ""
Write-Host "CSVs : $ResultsDir\batch_unet\  et  $ResultsDir\batch_segresnet\"
Write-Host ""
Write-Host "Prochaine etape : lance aggregate_5fold.py pour generer le tableau final"
Write-Host "  python scripts\kaggle\aggregate_5fold.py"
