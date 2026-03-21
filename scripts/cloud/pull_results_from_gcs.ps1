# PowerShell script to pull ggSwarm training results from GCS
# Usage: .\pull_results_from_gcs.ps1 -Family marl [-DryRun]

param(
    [ValidateSet('marl', 'hover')]
    [string]$Family = 'marl',
    [switch]$DryRun = $false
)

$GcsBucket = $env:GGSWARM_GCS_BUCKET ? $env:GGSWARM_GCS_BUCKET : "gs://gg-swarm-training-logs"
$LocalLogDir = "logs\skrl\ggswarm_$Family"
$RemotePath = "$GcsBucket/logs/skrl/ggswarm_$Family"

$RsyncArgs = @('-m', '-r', '-x', 'videos/.*')

if ($DryRun) {
    $RsyncArgs += '-n'
    Write-Host "[DRY-RUN] No files will be modified." -ForegroundColor Cyan
}

Write-Host "Pulling $RemotePath to $LocalLogDir (excluding videos)" -ForegroundColor Green
& gsutil @RsyncArgs rsync $RemotePath $LocalLogDir

if ($LASTEXITCODE -ne 0) {
    Write-Error "Pull failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "Pull complete." -ForegroundColor Green
