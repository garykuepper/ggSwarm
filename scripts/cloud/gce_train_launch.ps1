#requires -Version 5.1
<#
.SYNOPSIS
    Push local git, pull on GCE, and start detached training via train_and_push.sh.

.DESCRIPTION
    From the repo root on Windows (PowerShell): runs `git push`, then SSH to the
    VM to `git pull` and `nohup bash scripts/cloud/train_and_push.sh ...`.
    GCS URI is inlined per GCE ops rules (not sourced from ~/.bashrc).

.EXAMPLE
    .\scripts\cloud\gce_train_launch.ps1

.EXAMPLE
    .\scripts\cloud\gce_train_launch.ps1 hover-stability train --headless --max_iterations 92000

.EXAMPLE
    .\scripts\cloud\gce_train_launch.ps1 -SkipGitPush phase2 train --headless --max_iterations 120000

.NOTES
    Training tokens must not be passed as the first positional argument: PowerShell
    would bind them to -Branch. Either put switches first or use -Branch explicitly,
    e.g. .\gce_train_launch.ps1 -Branch main hover-stability train --headless ...
    This script uses the automatic ``$args`` list for all training tokens instead.
#>
param(
    [string]$Branch = "main",
    [string]$Instance = "isaacsim",
    [string]$Zone = "us-central1-a",
    [string]$Project = "gg-swarm",
    [switch]$SkipGitPush
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

# Use $args so the first token is never mistaken for -Branch (positional binding).
if (-not $args -or $args.Count -eq 0) {
    $trainTokens = @(
        "hover-stability", "train", "--headless", "--max_iterations", "80000"
    )
} else {
    $trainTokens = @($args)
}

$trainCmd = $trainTokens -join " "
# Backtick-escape $ so bash on the VM expands $(date ...), not PowerShell.
$remoteCmd = (
    "cd ~/ggSwarm && git pull origin $Branch && " +
    "GGSWARM_GCS_URI=gs://gg-swarm-training-logs nohup bash scripts/cloud/train_and_push.sh $trainCmd " +
    "> ~/train_ggswarm_`$(date +%Y%m%d_%H%M%S).log 2>&1 &"
)

Set-Location $repoRoot

if (-not $SkipGitPush) {
    Write-Host "[gce_train_launch] git push origin $Branch ..."
    git push origin $Branch
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed (exit $LASTEXITCODE)."
    }
} else {
    Write-Host "[gce_train_launch] -SkipGitPush: not running git push."
}

Write-Host "[gce_train_launch] SSH: git pull + train_and_push ($trainCmd) ..."
& gcloud compute ssh $Instance --zone=$Zone --project=$Project --command=$remoteCmd
if ($LASTEXITCODE -ne 0) {
    throw "gcloud compute ssh failed (exit $LASTEXITCODE)."
}

Write-Host "[gce_train_launch] Done. Tail log on VM: tail -f ~/train_ggswarm_*.log (latest)."
