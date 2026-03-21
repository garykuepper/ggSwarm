# PowerShell script to manually sync training results when automatic sync fails
# This is a workaround until VM scopes are properly configured
# Usage: .\manual_sync_results.ps1

param(
    [string]$Family = 'marl',
    [switch]$Force = $false
)

Write-Host "`n=== Manual Training Results Sync ===" -ForegroundColor Green

$GcsBucket = "gs://gg-swarm-training-logs"
$VmName = "isaacsim"
$Zone = "us-central1-a"
$LocalLogDir = "logs\skrl\ggswarm_$Family"
$RemotePath = "$GcsBucket/logs/skrl/ggswarm_$Family"

# Check if local logs already synced
$checkpointCount = (Get-ChildItem -Path $LocalLogDir -Recurse -Filter "*.pt" -ErrorAction SilentlyContinue | Measure-Object).Count

if ($checkpointCount -gt 0 -and -not $Force) {
    Write-Host "`n✓ Checkpoints already synced locally ($checkpointCount files found)" -ForegroundColor Green
    Write-Host "`nTo pull latest from GCS, run:" -ForegroundColor Yellow
    Write-Host "  .\scripts\cloud\pull_results_from_gcs.ps1 -Family $Family`n" -ForegroundColor Cyan
    exit 0
}

Write-Host "`nFalling back to direct VM copy via gcloud compute scp..." -ForegroundColor Yellow
Write-Host "(This bypasses GCS but gets the job done quickly)`n" -ForegroundColor Yellow

Write-Host "Copying logs from VM ($VmName) to local PC..." -ForegroundColor Cyan

$vmSourcePath = "/home/gkuep/ggSwarm/logs/skrl/ggswarm_$Family"

gcloud compute scp `
    --zone=$Zone `
    --recurse `
    "${VmName}:${vmSourcePath}" `
    $LocalLogDir `
    2>&1 | Select-Object -Last 10

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n❌ Copy failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n✓ Sync complete!" -ForegroundColor Green
Write-Host "`nCheckpoints are now available locally in: $LocalLogDir`n" -ForegroundColor Green

Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. List checkpoints: .\scripts\cloud\list_checkpoints.ps1 -Family $Family" -ForegroundColor Cyan
Write-Host "  2. Evaluate: python scripts/run.py $Family eval --checkpoint <path>" -ForegroundColor Cyan
Write-Host "  3. Play: python scripts/run.py $Family play --checkpoint <path> --video`n" -ForegroundColor Cyan
