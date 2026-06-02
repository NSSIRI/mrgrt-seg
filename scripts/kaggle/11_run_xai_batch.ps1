# =====================================================================
# Script tout-en-un XAI batch (5 folds x 2 modeles) :
#   1) Commit + push les fichiers XAI sur GitHub
#   2) Cree 2 batch notebooks XAI (1 par compte) via Kaggle CLI
#   3) Polle leur execution
#   4) Telecharge les CSV XAI (per_patient + adebayo)
#
# Pre-requis :
#   - Kaggle CLI installe (Python311\Scripts\kaggle.exe)
#   - access_token_compte1 et access_token_compte2 dans ~/.kaggle/
#   - scripts/run_xai_analysis.py, src/xai/*.py committe et push sur GitHub
#
# Usage (PowerShell, dans le dossier du projet) :
#   .\scripts\kaggle\11_run_xai_batch.ps1
# =====================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Users\Lenovo\Desktop\mrgrt_seg"
$Kaggle = "C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts\kaggle.exe"
$ResultsDir = Join-Path $ProjectRoot "results"
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Set-Location $ProjectRoot

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " BATCH XAI 5 folds x 2 modeles" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Access tokens (compte 1 et 2) ---------------------------------
$kaggleConf = "$env:USERPROFILE\.kaggle\access_token"
$kaggleConf1 = "$env:USERPROFILE\.kaggle\access_token_compte1"
$kaggleConf2 = "$env:USERPROFILE\.kaggle\access_token_compte2"

if (-not (Test-Path $kaggleConf1)) {
    Write-Host "[ERREUR] $kaggleConf1 absent" -ForegroundColor Red
    Write-Host "  Backup compte 1 avec : Copy-Item $kaggleConf $kaggleConf1"
    exit 1
}
if (-not (Test-Path $kaggleConf2)) {
    Write-Host "[ATTENTION] $kaggleConf2 absent" -ForegroundColor Yellow
    $ans = Read-Host "Continuer avec compte 1 seulement (UNet) ? (O/N)"
    if ($ans -ne "O" -and $ans -ne "o") { exit 0 }
    $SkipCompte2 = $true
} else {
    $SkipCompte2 = $false
}

# --- 0. Push GitHub si besoin --------------------------------------
Write-Host "`n[0] Verification GitHub..." -ForegroundColor Yellow
$status = git status --porcelain scripts/run_xai_analysis.py scripts/kaggle/xai_kaggle.py scripts/kaggle/make_xai_notebook.py src/xai/ 2>&1
if ($status) {
    Write-Host "  Changements detectes, commit + push..."
    git add scripts/run_xai_analysis.py scripts/kaggle/xai_kaggle.py scripts/kaggle/make_xai_notebook.py src/xai/
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    git commit -m "XAI : center-crop fix + scripts" 2>&1 | Out-Null
    $pushOut = & git push 2>&1 | Out-String
    $ErrorActionPreference = $prev
    Write-Host "  $($pushOut.Trim())"
} else {
    Write-Host "  Pas de changements locaux. Push global pour s'assurer..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $pushOut = & git push 2>&1 | Out-String
    $ErrorActionPreference = $prev
    Write-Host "  $($pushOut.Trim())"
}

# --- Helper : push + poll un kernel XAI ----------------------------
function Invoke-KaggleXaiBatch {
    param(
        [string]$Username,
        [string]$Model,
        [string]$TokenSourcePath
    )

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Yellow
    Write-Host " $($Model.ToUpper()) XAI batch ($Username)" -ForegroundColor Yellow
    Write-Host "============================================" -ForegroundColor Yellow

    Copy-Item $TokenSourcePath $kaggleConf -Force

    $kernelId = "$Username/mrgrt-xai-$Model-batch"
    $tempDir = Join-Path $env:TEMP "kaggle_xai_$Model"
    if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
    New-Item -ItemType Directory -Path $tempDir | Out-Null

    $makeNb = Join-Path $ProjectRoot "scripts\kaggle\make_xai_notebook.py"
    python $makeNb --model $Model --username $Username --out_dir $tempDir 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] generation notebook a echoue" -ForegroundColor Red
        return $false
    }

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

    Write-Host "Polling toutes les 60s (max 90 min)..."
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
        if ($elapsed -gt 90) {
            Write-Host "[TIMEOUT] 90 min depasses" -ForegroundColor Yellow
            return $false
        }
    }

    Write-Host "Telechargement outputs..."
    $outDir = Join-Path $ResultsDir "xai_${Model}"
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    New-Item -ItemType Directory -Path $outDir | Out-Null
    & $Kaggle kernels output $kernelId -p $outDir 2>&1 | Out-Null
    Write-Host "Outputs : $outDir"

    return $true
}

# --- Run UNet XAI (compte 1) ----------------------------------------
$unetOK = Invoke-KaggleXaiBatch -Username "abdelhalimnssiri" -Model "unet" -TokenSourcePath $kaggleConf1

# --- Run SegResNet XAI (compte 2) -----------------------------------
$segOK = $false
if (-not $SkipCompte2) {
    $segOK = Invoke-KaggleXaiBatch -Username "nssiri02" -Model "segresnet" -TokenSourcePath $kaggleConf2
}

# --- Restore compte 1 -----------------------------------------------
Copy-Item $kaggleConf1 $kaggleConf -Force
Write-Host ""
Write-Host "access_token restaure (compte 1)"

# --- Bilan ----------------------------------------------------------
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host " XAI BATCH TERMINE" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "UNet      : $(if ($unetOK) {'OK'} else {'ECHEC'})"
Write-Host "SegResNet : $(if ($segOK) {'OK'} elseif ($SkipCompte2) {'SKIP'} else {'ECHEC'})"
Write-Host ""
Write-Host "CSVs XAI :"
Write-Host "  $ResultsDir\xai_unet\"
Write-Host "  $ResultsDir\xai_segresnet\"
Write-Host ""
Write-Host "Liste des CSV produits :"
Get-ChildItem -Path $ResultsDir -Recurse -Filter "*xai_per_patient.csv" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.FullName)"
}
Get-ChildItem -Path $ResultsDir -Recurse -Filter "*adebayo.csv" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $($_.FullName)"
}
