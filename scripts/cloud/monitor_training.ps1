# PowerShell script to SSH tunnel to the GCE VM and open TensorBoard in a local browser
# Usage: .\monitor_training.ps1 [-Family marl|hover] [-Port 6006] [-Zone us-central1-a] [-Instance isaacsim]

param(
    [ValidateSet('marl', 'hover')]
    [string]$Family = 'marl',
    [int]$Port = 6006,
    [string]$Zone = 'us-central1-a',
    [string]$Instance = 'isaacsim'
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  TensorBoard Remote Monitor Setup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

Write-Host "Opening SSH tunnel to $Instance ($Zone) on port $Port..." -ForegroundColor Green
Write-Host "Family: $Family`n" -ForegroundColor Yellow

# Inform the user before starting
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "  1. Keep this terminal open while monitoring TensorBoard"
Write-Host "  2. Your browser will open to http://localhost:$Port"
Write-Host "  3. Press Ctrl+C here to close the tunnel when done`n" -ForegroundColor Yellow

# Give a brief pause for the user to read the instructions
Start-Sleep -Seconds 1

# Open the SSH tunnel in the background
# -N: don't execute a remote command
# -L 6006:127.0.0.1:6006: forward local port to remote port
Write-Host "Starting tunnel..." -ForegroundColor Cyan
& gcloud compute ssh $Instance `
    --zone=$Zone `
    -- -N -L "${Port}:127.0.0.1:6006" `
    2>&1 | ForEach-Object {
        if ($_ -match "error|Error|ERROR") {
            Write-Host $_ -ForegroundColor Red
        } else {
            Write-Host $_ -ForegroundColor Gray
        }
    } &

$tunnelPid = $?
Write-Host "Tunnel started (PID: $tunnelPid)`n" -ForegroundColor Green

# Give the tunnel a moment to establish
Start-Sleep -Seconds 2

# Open the browser to localhost:Port
$url = "http://localhost:$Port"
Write-Host "Opening browser to $url..." -ForegroundColor Green
Start-Process $url

Write-Host "`nTensorBoard is loading at $url" -ForegroundColor Green
Write-Host "Keep this terminal open. Press Ctrl+C to stop the tunnel.`n" -ForegroundColor Yellow

# Keep the script running so the tunnel stays open
while ($true) {
    Start-Sleep -Seconds 1
}
