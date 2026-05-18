[CmdletBinding()]
param(
    [switch]$Fix,
    [switch]$SkipAudit,
    [switch]$SkipBandit
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "=== $Message ===" -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Message)
    Write-Host "--> $Message" -ForegroundColor Yellow
}

function Resolve-RepoRoot {
    $scriptDir = $PSScriptRoot
    return (Resolve-Path (Join-Path $scriptDir "..")).Path
}

function Get-CommandPath {
    param(
        [string]$PreferredPath,
        [string]$FallbackCommand
    )

    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return $PreferredPath
    }

    $fallback = Get-Command $FallbackCommand -ErrorAction SilentlyContinue
    if ($fallback) {
        return $fallback.Source
    }

    throw "Commande introuvable : $FallbackCommand"
}

function Try-GetCommandPath {
    param(
        [string]$PreferredPath,
        [string]$FallbackCommand
    )

    if ($PreferredPath -and (Test-Path $PreferredPath)) {
        return $PreferredPath
    }

    $fallback = Get-Command $FallbackCommand -ErrorAction SilentlyContinue
    if ($fallback) {
        return $fallback.Source
    }

    return $null
}

function Invoke-External {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Echec de la commande : $FilePath $($Arguments -join ' ')"
    }
}

function Test-ToolInstalled {
    param([string]$CommandName)
    return [bool](Get-Command $CommandName -ErrorAction SilentlyContinue)
}

$repoRoot = Resolve-RepoRoot
Set-Location $repoRoot

Write-Section "Workflow local pre-push - Projet ML"
Write-Host "Repository : $repoRoot"
Write-Host "Mode correction : $Fix"

# Configuration des chemins du venv
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$venvRuff = Join-Path $repoRoot ".venv\Scripts\ruff.exe"
$venvBlack = Join-Path $repoRoot ".venv\Scripts\black.exe"
$venvPytest = Join-Path $repoRoot ".venv\Scripts\pytest.exe"
$venvBandit = Join-Path $repoRoot ".venv\Scripts\bandit.exe"
$venvPipAudit = Join-Path $repoRoot ".venv\Scripts\pip-audit.exe"

# Resolution des commandes
$pythonCmd = Get-CommandPath -PreferredPath $venvPython -FallbackCommand "python"
$ruffCmd = Get-CommandPath -PreferredPath $venvRuff -FallbackCommand "ruff"
$blackCmd = Get-CommandPath -PreferredPath $venvBlack -FallbackCommand "black"
$pytestCmd = Get-CommandPath -PreferredPath $venvPytest -FallbackCommand "pytest"

if (-not $SkipBandit) {
    $banditCmd = Try-GetCommandPath -PreferredPath $venvBandit -FallbackCommand "bandit"
}

if (-not $SkipAudit) {
    $pipAuditCmd = Try-GetCommandPath -PreferredPath $venvPipAudit -FallbackCommand "pip-audit"
}

Write-Section "Preparation"
Write-Step "Verification des dossiers d'artefacts attendus par les tests"
New-Item -ItemType Directory -Force artifacts\models | Out-Null
New-Item -ItemType Directory -Force artifacts\metrics | Out-Null
New-Item -ItemType Directory -Force artifacts\preprocessors | Out-Null
New-Item -ItemType Directory -Force artifacts\predictions | Out-Null

if ($Fix) {
    Write-Section "Corrections automatiques"
    Write-Step "Ruff auto-fix"
    Invoke-External -FilePath $ruffCmd -Arguments @("check", ".", "--fix")

    Write-Step "Ruff format"
    Invoke-External -FilePath $ruffCmd -Arguments @("format", ".")

    Write-Step "Black format"
    Invoke-External -FilePath $blackCmd -Arguments @(".")
}

Write-Section "Controles Python"

Write-Step "Ruff lint"
Invoke-External -FilePath $ruffCmd -Arguments @("check", ".")

Write-Step "Ruff format check"
Invoke-External -FilePath $ruffCmd -Arguments @("format", "--check", ".")

Write-Step "Black format check"
Invoke-External -FilePath $blackCmd -Arguments @("--check", ".")
$pytestArgs = @("-v", "--tb=short")

Write-Step "Pytest avec couverture"
$pytestArgs = @("-v", "--tb=short")
if (Get-Command pytest -ErrorAction SilentlyContinue) {
    $pytestArgs += @("--cov=src", "--cov-report=term-missing")
}
Invoke-External -FilePath $pytestCmd -Arguments $pytestArgs

if (-not $SkipBandit) {
    if ($banditCmd) {
        Write-Step "Bandit security scan"
        Invoke-External -FilePath $banditCmd -Arguments @("-r", "src/")
    }
    else {
        Write-Host "Bandit non installe localement : scan saute." -ForegroundColor DarkYellow
    }
}
else {
    Write-Host "Bandit ignore a la demande." -ForegroundColor DarkYellow
}

if (-not $SkipAudit) {
    if ($pipAuditCmd) {
        Write-Step "pip-audit dependency scan"
        Invoke-External -FilePath $pipAuditCmd -Arguments @()
    }
    else {
        Write-Host "pip-audit non installe localement : scan saute." -ForegroundColor DarkYellow
    }
}
else {
    Write-Host "pip-audit ignore a la demande." -ForegroundColor DarkYellow
}

Write-Section "Termine"
Write-Host "Tous les controles demandes sont passes avec succes." -ForegroundColor Green
Write-Host "Tu peux push en confiance." -ForegroundColor Green