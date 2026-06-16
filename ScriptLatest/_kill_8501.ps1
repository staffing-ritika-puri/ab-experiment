$pids = Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
foreach ($p in $pids) {
    try {
        Stop-Process -Id $p -Force -ErrorAction Stop
        Write-Host "killed $p"
    } catch {
        Write-Host "could not kill $p"
    }
}
Start-Sleep -Seconds 1
if (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue) {
    Write-Host "still in use"
} else {
    Write-Host "port 8501 free"
}
