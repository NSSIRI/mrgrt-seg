# =====================================================================
# Script tout-en-un pour evaluer UNet fold 0 :
#   1) Commit + push le fix evaluate.py vers GitHub
#   2) Relance le notebook d'evaluation Kaggle
#   3) Polle le statut jusqu'a COMPLETE
#   4) Telecharge les outputs
#   5) Affiche les metriques par OAR
#
# Usage (depuis PowerShell, dans le dossier du projet) :
#   .\scripts\kaggle\09_run_eval_unet_fold0.ps1
#
# Pre-requis :
#   - Kaggle CLI installe (Python311\Scripts\kaggle.exe)
#   - kaggle.json du compte 1 actif (~/.kaggle/kaggle.json)
#   - Notebook d'evaluation deja cree sur Kaggle :
#       abdelhalimnssiri/mrgrt-eval-unet-fold0
#     (si different, modifier $EVAL_KERNEL ci-dessous)
# =====================================================================

$ErrorActionPreference = "Stop"

# --- Configuration ---------------------------------------------------
$ProjectRoot = "C:\Users\Lenovo\Desktop\mrgrt_seg"
$Kaggle = "C:\Users\Lenovo\AppData\Roaming\Python\Python311\Scripts\kaggle.exe"
$EvalKernel = "abdelhalimnssiri/mrgrt-eval-unet-fold0"
$ResultsDir = Join-Path $ProjectRoot "results\fold0_unet_eval"
$PollIntervalSec = 60

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " UNet fold 0 - Evaluation tout-en-un" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan

# --- Sanity checks ---------------------------------------------------
if (-not (Test-Path $Kaggle)) {
    Write-Host "[ERREUR] Kaggle CLI introuvable : $Kaggle" -ForegroundColor Red
    Write-Host "  Installation : pip install kaggle --user"
    exit 1
}
Set-Location $ProjectRoot
if (-not (Test-Path "scripts\evaluate.py")) {
    Write-Host "[ERREUR] Pas dans le dossier du projet ?" -ForegroundColor Red
    exit 1
}

# --- 1. Commit + push fix evaluate.py --------------------------------
Write-Host "`n[etape 1/5] Commit + push fix evaluate.py..." -ForegroundColor Yellow

# Verifier que le fix has_cucim=False est bien dans le fichier
$grep = Select-String -Path "scripts\evaluate.py" -Pattern "has_cucim\s*=\s*False" -SimpleMatch:$false
if (-not $grep) {
    Write-Host "[ATTENTION] Fix 'has_cucim = False' ABSENT de evaluate.py" -ForegroundColor Yellow
    Write-Host "  Ouvre scripts\evaluate.py et verifie qu'il contient :"
    Write-Host "  import monai.metrics.utils as _mmu"
    Write-Host "  _mmu.has_cucim = False"
    $ans = Read-Host "Continuer quand meme ? (O/N)"
    if ($ans -ne "O" -and $ans -ne "o") { exit 0 }
} else {
    Write-Host "  [OK] Fix has_cucim = False present"
}

# Verifier si scripts/evaluate.py a des changes a committer
$gitStatus = git diff --name-only scripts/evaluate.py
if ($gitStatus) {
    git add scripts/evaluate.py
    try {
        git commit -m "Fix: disable cuCIM in MONAI eval (NVRTC C++17 error on Kaggle cu128)" 2>&1 | Out-Null
        Write-Host "  [OK] Commit cree"
    } catch {
        Write-Host "  [INFO] Rien a committer ou commit deja fait"
    }
} else {
    Write-Host "  [INFO] scripts/evaluate.py n'a pas de changement local"
}

# Push (ignore "Everything up-to-date" qui sort en stderr)
Write-Host "  Push vers GitHub..."
$prev = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$pushOutput = & git push 2>&1 | Out-String
$ErrorActionPreference = $prev
Write-Host "  $($pushOutput.Trim())"

# --- 2. Verifier que le kernel d'eval existe -------------------------
Write-Host "`n[etape 2/5] Verification du kernel d'evaluation..." -ForegroundColor Yellow
$statusOut = & $Kaggle kernels status $EvalKernel 2>&1
if ($statusOut -match "404|not found|Permission") {
    Write-Host "[ERREUR] Kernel $EvalKernel introuvable" -ForegroundColor Red
    Write-Host "  Va sur https://www.kaggle.com/code et cree un notebook nomme 'mrgrt-eval-unet-fold0'"
    Write-Host "  avec le script d'evaluation (cf scripts/kaggle/evaluate_kaggle.ipynb)"
    exit 1
}
Write-Host "  Statut actuel : $statusOut"

# --- 3. Relancer le kernel ------------------------------------------
Write-Host "`n[etape 3/5] Relance du notebook d'evaluation..." -ForegroundColor Yellow

# Pull la version actuelle pour avoir le metadata + code
$tempDir = Join-Path $env:TEMP "kaggle_eval_relaunch"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir | Out-Null

& $Kaggle kernels pull $EvalKernel -p $tempDir -m 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERREUR] kernels pull a echoue" -ForegroundColor Red
    exit 1
}

# Push pour relancer (re-execute toutes les cellules)
Push-Location $tempDir
try {
    Write-Host "  Push pour declencher un nouveau run..."
    & $Kaggle kernels push -p . 2>&1 | Tee-Object -Variable pushOutput | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] kernels push a echoue : $pushOutput" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Run lance : https://www.kaggle.com/code/$EvalKernel"
} finally {
    Pop-Location
}

# --- 4. Polling jusqu'a fin -----------------------------------------
Write-Host "`n[etape 4/5] Surveillance du run (poll toutes les ${PollIntervalSec}s)..." -ForegroundColor Yellow

$start = Get-Date
$maxWaitMinutes = 30  # eval = inference seule, devrait etre <15 min

while ($true) {
    Start-Sleep -Seconds $PollIntervalSec
    $elapsed = ((Get-Date) - $start).TotalMinutes
    $st = & $Kaggle kernels status $EvalKernel 2>&1
    Write-Host ("  [{0:F1} min] {1}" -f $elapsed, $st)

    if ($st -match "COMPLETE") {
        Write-Host "  [OK] Run termine avec succes" -ForegroundColor Green
        break
    } elseif ($st -match "ERROR|CANCELLED|FAILED") {
        Write-Host "  [ERREUR] Run a echoue : $st" -ForegroundColor Red
        Write-Host "  Verifier les logs : https://www.kaggle.com/code/$EvalKernel"
        exit 1
    } elseif ($elapsed -gt $maxWaitMinutes) {
        Write-Host "  [TIMEOUT] $maxWaitMinutes min depasses, on arrete le polling" -ForegroundColor Yellow
        Write-Host "  Verifier manuellement : https://www.kaggle.com/code/$EvalKernel"
        exit 1
    }
}

# --- 5. Telecharger + afficher metriques ----------------------------
Write-Host "`n[etape 5/5] Telechargement des outputs..." -ForegroundColor Yellow
if (Test-Path $ResultsDir) {
    Remove-Item -Recurse -Force $ResultsDir
}
New-Item -ItemType Directory -Path $ResultsDir | Out-Null

& $Kaggle kernels output $EvalKernel -p $ResultsDir 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ATTENTION] Telechargement partiel" -ForegroundColor Yellow
}

# Chercher le CSV de metriques
$csv = Get-ChildItem -Path $ResultsDir -Recurse -Filter "*metrics*.csv" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $csv) {
    Write-Host "[ATTENTION] CSV de metriques introuvable dans $ResultsDir" -ForegroundColor Yellow
    Write-Host "  Liste des fichiers telecharges :"
    Get-ChildItem -Path $ResultsDir -Recurse | Select-Object Name, @{N='KB';E={[math]::Round($_.Length/1KB,1)}} | Format-Table
    exit 1
}

Write-Host "  CSV trouve : $($csv.FullName)" -ForegroundColor Green

# --- Affichage des moyennes par OAR ---------------------------------
Write-Host "`n============================================" -ForegroundColor Green
Write-Host " RESULTATS UNet fold 0" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

$rows = Import-Csv $csv.FullName
$n = $rows.Count
Write-Host "Patients evalues : $n`n"

$organs = @("poumon_g", "poumon_d", "coeur", "oesophage")
$metrics = @("dsc", "iou", "hd95", "sdsc")

foreach ($m in $metrics) {
    Write-Host "--- $($m.ToUpper()) ---"
    foreach ($o in $organs) {
        $col = "${m}_${o}"
        if ($rows[0].PSObject.Properties.Name -contains $col) {
            $vals = $rows | ForEach-Object {
                $v = $_.$col
                if ($v -and $v -ne "" -and $v -ne "nan") { [double]$v }
            }
            if ($vals.Count -gt 0) {
                $mean = ($vals | Measure-Object -Average).Average
                $std = [math]::Sqrt((($vals | ForEach-Object { [math]::Pow($_ - $mean, 2) }) | Measure-Object -Sum).Sum / $vals.Count)
                Write-Host ("  {0,-11} {1,6:F3} +/- {2,5:F3}" -f $o, $mean, $std)
            } else {
                Write-Host ("  {0,-11} (aucune valeur valide)" -f $o) -ForegroundColor DarkGray
            }
        }
    }
    Write-Host ""
}

# DSC moyen global
$allDsc = @()
foreach ($o in $organs) {
    $col = "dsc_${o}"
    if ($rows[0].PSObject.Properties.Name -contains $col) {
        $allDsc += $rows | ForEach-Object {
            $v = $_.$col
            if ($v -and $v -ne "" -and $v -ne "nan") { [double]$v }
        }
    }
}
if ($allDsc.Count -gt 0) {
    $globalMean = ($allDsc | Measure-Object -Average).Average
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host (" DSC MOYEN GLOBAL : {0:F4}" -f $globalMean) -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
}

Write-Host "`nCSV complet : $($csv.FullName)"
Write-Host "Ouvrir avec : Import-Csv `"$($csv.FullName)`" | Out-GridView"
