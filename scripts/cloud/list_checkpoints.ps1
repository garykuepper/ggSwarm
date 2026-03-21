# PowerShell script to list available checkpoints and select one for playback
# Usage: .\list_checkpoints.ps1 [-Family marl|hover] [-Latest N]

param(
    [ValidateSet('marl', 'hover')]
    [string]$Family = 'marl',
    [int]$Latest = 10
)

$LogDir = "logs\skrl\ggswarm_$Family"

if (-not (Test-Path $LogDir)) {
    Write-Error "Log directory not found: $LogDir"
    exit 1
}

# Find all checkpoint files
$Checkpoints = @()

foreach ($RunDir in (Get-ChildItem $LogDir -Directory | Sort-Object LastWriteTime -Descending)) {
    $CheckpointsDir = Join-Path $RunDir.FullName "checkpoints"
    if (Test-Path $CheckpointsDir) {
        foreach ($Ckpt in (Get-ChildItem $CheckpointsDir -Filter "*.pt")) {
            $IsBest = $Ckpt.Name -eq "best_agent.pt"
            $SizeMB = [math]::Round($Ckpt.Length / 1MB, 1)
            $Checkpoints += @{
                Path = (Join-Path $RunDir.Name "checkpoints" $Ckpt.Name)
                Name = $Ckpt.Name
                RunDir = $RunDir.Name
                Date = $RunDir.LastWriteTime
                Size = $SizeMB
                IsBest = $IsBest
            }
        }
    }
}

if ($Checkpoints.Count -eq 0) {
    Write-Host "No checkpoints found in $LogDir"
    exit 0
}

# Sort by date (newest first) and limit to Latest N
$Checkpoints = $Checkpoints | Sort-Object { $_.Date } -Descending | Select-Object -First $Latest

Write-Host "`nAvailable checkpoints (newest first):`n" -ForegroundColor Green

for ($i = 0; $i -lt $Checkpoints.Count; $i++) {
    $Ckpt = $Checkpoints[$i]
    $BestFlag = if ($Ckpt.IsBest) { " [BEST]" } else { "" }
    Write-Host "[$($i+1)] $($Ckpt.Path) ($($Ckpt.Size) MB)$BestFlag" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "To use a checkpoint, run:" -ForegroundColor Yellow
Write-Host "  python scripts/run.py phase2 play --checkpoint `"logs\skrl\ggswarm_$Family\<selected_checkpoint>`"" -ForegroundColor Yellow
Write-Host ""

# Print the latest checkpoint path for easy use
$LatestPath = Join-Path $LogDir $Checkpoints[0].Path
Write-Host "Latest checkpoint:" -ForegroundColor Green
Write-Host "  $LatestPath" -ForegroundColor Cyan
