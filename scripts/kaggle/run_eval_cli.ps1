# =====================================================================
# run_eval_cli.ps1 — Lance l'evaluation Kaggle depuis le PC via l'API CLI.
#
# Pousse un notebook d'evaluation, l'execute sur GPU, et telecharge le CSV.
# A executer dans PowerShell sur le PC (pas sur Kaggle).
#
# PRE-REQUIS :
#   1. pip install kaggle  (et ~/.kaggle/kaggle.json configure pour le COMPTE cible)
#   2. Le dataset mrgrt-oar-thorax-clean-v2 existe sur ce compte
#   3. Le notebook d'entrainement (avec best.pt) existe sur ce compte
#   4. Repo GitHub PUBLIC  (sinon, voir note secret en bas)
#
# Usage :
#   .\scripts\kaggle\run_eval_cli.ps1 -Model segresnet -Fold 0 `
#       -KaggleUser "ton_user_kaggle" `
#       -TrainKernel "ton_user_kaggle/mrgrt-train-segresnet-fold0"
# =====================================================================
param(
    [Parameter(Mandatory=$true)][string]$Model,        # unet | segresnet
    [Parameter(Mandatory=$true)][int]$Fold,
    [Parameter(Mandatory=$true)][string]$KaggleUser,   # username du compte cible
    [Parameter(Mandatory=$true)][string]$TrainKernel,  # user/slug du notebook training (source du best.pt)
    [string]$Dataset = "",                              # defaut: <KaggleUser>/mrgrt-oar-thorax-clean-v2
    [string]$GithubUser = "NSSIRI",
    [switch]$RepoPublic = $true                         # si repo public, pas de secret necessaire
)

$ErrorActionPreference = "Stop"
if (-not $Dataset) { $Dataset = "$KaggleUser/mrgrt-oar-thorax-clean-v2" }

# 1) Verifier kaggle CLI
Write-Host "[1] Verification kaggle CLI..."
kaggle --version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "kaggle CLI absent. Faire: pip install kaggle" }

# 2) Dossier de staging du kernel
$slug = "mrgrt-eval-$Model-fold$Fold"
$stage = Join-Path $env:TEMP $slug
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

# 3) Copier le notebook d'eval
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$nb = Join-Path $repoRoot "scripts\kaggle\evaluate_kaggle.ipynb"
if (-not (Test-Path $nb)) { throw "Notebook introuvable: $nb" }
Copy-Item $nb (Join-Path $stage "evaluate_kaggle.ipynb")

# 4) Generer kernel-metadata.json
$secretMode = if ($RepoPublic) { "None" } else { "`"GITHUB_TOKEN`"" }
$meta = @{
    id              = "$KaggleUser/$slug"
    title           = $slug
    code_file       = "evaluate_kaggle.ipynb"
    language        = "python"
    kernel_type     = "notebook"
    is_private      = $true
    enable_gpu      = $true
    enable_internet = $true
    dataset_sources = @($Dataset)
    kernel_sources  = @($TrainKernel)
    competition_sources = @()
} | ConvertTo-Json -Depth 5
$meta | Out-File -FilePath (Join-Path $stage "kernel-metadata.json") -Encoding utf8
Write-Host "[2] Metadata genere pour $KaggleUser/$slug"
Write-Host "    Dataset : $Dataset"
Write-Host "    Source best.pt : $TrainKernel"

# 5) Push + run
Write-Host "[3] Push du kernel (lance l'execution sur Kaggle)..."
kaggle kernels push -p $stage
if ($LASTEXITCODE -ne 0) { throw "Echec kaggle kernels push" }

# 6) Polling du statut
Write-Host "[4] Attente de la fin d'execution (polling toutes les 30s)..."
do {
    Start-Sleep -Seconds 30
    $status = kaggle kernels status "$KaggleUser/$slug" 2>&1 | Out-String
    Write-Host "    $status".Trim()
} while ($status -match "running|queued")

# 7) Telecharger l'output
$outDir = Join-Path $repoRoot "results\$slug"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Write-Host "[5] Telechargement de l'output dans $outDir"
kaggle kernels output "$KaggleUser/$slug" -p $outDir

Write-Host ""
Write-Host "=== TERMINE ==="
Write-Host "Resultats dans : $outDir"
Write-Host "Cherche le CSV : ${Model}_fold${Fold}_metrics.csv"
Get-ChildItem $outDir -Recurse -Filter "*.csv" | ForEach-Object { Write-Host "  -> $($_.FullName)" }
