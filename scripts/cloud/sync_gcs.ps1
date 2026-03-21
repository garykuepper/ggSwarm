# PowerShell script for syncing ggSwarm logs to/from GCS.
# Usage:
#   .\sync_gcs.ps1 push --family marl
#   .\sync_gcs.ps1 pull --family marl --include-videos
#   .\sync_gcs.ps1 pull --family hover -n (dry-run)

param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('push', 'pull')]
    [string]$Action,

    [Parameter(Mandatory=$false)]
    [ValidateSet('marl', 'hover')]
    [string]$Family = 'marl',

    [Parameter(Mandatory=$false)]
    [switch]$IncludeVideos = $false,

    [Parameter(Mandatory=$false)]
    [switch]$DryRun = $false
)

# Read GCS URI from environment; fall back to error if not set.
$GcsUri = $env:GGSWARM_GCS_URI
if (-not $GcsUri) {
    Write-Error "Environment variable GGSWARM_GCS_URI not set. Set it to gs://bucket-name/prefix and retry."
    exit 1
}

# Determine local and remote paths based on family (marl or hover).
$LocalLogDir = if ($Family -eq 'marl') {
    "logs\skrl\ggswarm_marl"
} else {
    "logs\skrl\ggswarm_hover"
}

$RemotePath = "$GcsUri/logs/skrl/ggswarm_$Family"

# Construct rsync arguments.
$RsyncArgs = @('-m', '-r')

# Add video exclusion unless --include-videos was passed.
if (-not $IncludeVideos) {
    $RsyncArgs += '-x', 'videos/.*'
}

# Add dry-run flag if requested.
if ($DryRun) {
    $RsyncArgs += '-n'
    Write-Host "[DRY-RUN] No files will be modified." -ForegroundColor Cyan
}

# Perform push or pull.
if ($Action -eq 'push') {
    Write-Host "Pushing $LocalLogDir to $RemotePath" -ForegroundColor Green
    & gsutil @RsyncArgs rsync $LocalLogDir $RemotePath
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Push failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "Push complete." -ForegroundColor Green
} else {
    Write-Host "Pulling $RemotePath to $LocalLogDir" -ForegroundColor Green
    & gsutil @RsyncArgs rsync $RemotePath $LocalLogDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Pull failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
    Write-Host "Pull complete." -ForegroundColor Green
}
