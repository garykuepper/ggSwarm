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
    Do not add string parameters before the training tokens: PowerShell will bind the
    first positional string to them (e.g. ``hover-stability`` -> wrong git branch).
    Git branch and SSH target use defaults or env: GGSWARM_GIT_BRANCH, GGSWARM_GCE_INSTANCE,
    GGSWARM_GCE_ZONE, GGSWARM_GCP_PROJECT.
#>
param(
    [switch]$SkipGitPush,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs = @()
)

$ErrorActionPreference = "Stop"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

$Branch = if ($env:GGSWARM_GIT_BRANCH) { $env:GGSWARM_GIT_BRANCH } else { "main" }
$Instance = if ($env:GGSWARM_GCE_INSTANCE) { $env:GGSWARM_GCE_INSTANCE } else { "isaacsim" }
$Zone = if ($env:GGSWARM_GCE_ZONE) { $env:GGSWARM_GCE_ZONE } else { "us-central1-a" }
$Project = if ($env:GGSWARM_GCP_PROJECT) { $env:GGSWARM_GCP_PROJECT } else { "gg-swarm" }

if (-not $TrainArgs -or $TrainArgs.Count -eq 0) {
    $trainTokens = @(
        "hover-stability", "train", "--headless", "--max_iterations", "80000"
    )
} else {
    $trainTokens = $TrainArgs
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
