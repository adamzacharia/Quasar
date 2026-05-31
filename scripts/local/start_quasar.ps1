param(
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 3001,
    [int]$BackendTimeoutSeconds = 180,
    [int]$FrontendTimeoutSeconds = 120,
    [switch]$Restart
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$UiDir = Join-Path $RepoRoot "ui-pro"
$BackendLog = Join-Path $UiDir "uvicorn.local.log"
$FrontendLog = Join-Path $UiDir "next.local.log"

function Get-ListenerProcess {
    param([int]$Port)
    $conn = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $conn) { return $null }
    return Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
}

function Stop-QuasarDevProcess {
    param([int]$Port)
    $proc = Get-ListenerProcess -Port $Port
    if (-not $proc) { return }

    $cmd = [string]$proc.CommandLine
    if ($cmd -notmatch "launch\.py|uvicorn|next dev|next\\dist\\server\\lib\\start-server\.js|npm-cli\.js|npm\.cmd") {
        throw "Port $Port is in use by PID $($proc.ProcessId), but it does not look like a Quasar dev process. Stop it manually or choose another port."
    }

    Write-Host "Stopping existing Quasar dev process on port $Port (PID $($proc.ProcessId))..."
    Stop-Process -Id $proc.ProcessId -Force
    Start-Sleep -Seconds 2
}

function Assert-PortFree {
    param([int]$Port)
    $proc = Get-ListenerProcess -Port $Port
    if (-not $proc) { return }

    if ($Restart) {
        Stop-QuasarDevProcess -Port $Port
        return
    }

    throw "Port $Port is already in use by PID $($proc.ProcessId). Re-run with -Restart or choose another port."
}

function Resolve-Python {
    $condaPython = Join-Path $env:USERPROFILE "anaconda3\envs\quasar\python.exe"
    $venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"

    foreach ($candidate in @($condaPython, $venvPython, "python")) {
        try {
            $check = & $candidate -c "import uvicorn; import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $check) {
                return $candidate
            }
        } catch {
            continue
        }
    }

    throw "Could not find a Python interpreter with uvicorn installed. Install backend requirements or use the quasar Conda env."
}

function Wait-ForPort {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
            Write-Host "$Name is listening on port $Port."
            return
        }
        Start-Sleep -Seconds 1
    }

    throw "$Name did not start on port $Port within $TimeoutSeconds seconds."
}

Assert-PortFree -Port $BackendPort
Assert-PortFree -Port $FrontendPort

$PythonExe = Resolve-Python

$backendCommand = @"
`$env:QUASAR_ENV='development'
`$env:ENVIRONMENT='development'
`$env:APP_ENV='development'
`$env:QUASAR_FORCE_LOCAL_DB='1'
`$env:QUASAR_ENABLE_LOCAL_TEST_LOGIN='1'
`$env:QUASAR_LOCAL_TEST_USERNAME='1@1'
`$env:QUASAR_LOCAL_TEST_EMAIL='1@1'
`$env:QUASAR_LOCAL_TEST_DISPLAY_NAME='1'
`$env:QUASAR_LOCAL_TEST_PASSWORD='1'
`$env:TURSO_DATABASE_URL=''
`$env:TURSO_AUTH_TOKEN=''
`$env:PORT='$BackendPort'
& '$PythonExe' launch.py *> '$BackendLog'
"@

$frontendCommand = @"
`$env:NEXT_PUBLIC_API_URL='http://localhost:$BackendPort'
npm.cmd run dev -- -p $FrontendPort *> '$FrontendLog'
"@

Write-Host "Starting Quasar backend on http://localhost:$BackendPort ..."
Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand) `
    -WorkingDirectory $UiDir `
    -WindowStyle Hidden

Write-Host "Starting Quasar frontend on http://localhost:$FrontendPort ..."
Start-Process -FilePath powershell.exe `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $frontendCommand) `
    -WorkingDirectory $UiDir `
    -WindowStyle Hidden

Wait-ForPort -Port $BackendPort -Name "Backend" -TimeoutSeconds $BackendTimeoutSeconds
Wait-ForPort -Port $FrontendPort -Name "Frontend" -TimeoutSeconds $FrontendTimeoutSeconds

Write-Host ""
Write-Host "Quasar local stack is ready:"
Write-Host "  Frontend: http://localhost:$FrontendPort"
Write-Host "  Backend:  http://localhost:$BackendPort"
Write-Host "  Login:    1@1 / 1"
Write-Host ""
Write-Host "Logs:"
Write-Host "  Backend:  $BackendLog"
Write-Host "  Frontend: $FrontendLog"
