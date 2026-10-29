#Requires -Version 5.1
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Backend = Join-Path $Root "backend"
$Frontend = Join-Path $Root "frontend"
$Python = Join-Path $Backend ".venv\Scripts\python.exe"
$EnvFile = Join-Path $Backend ".env"
$EnvExample = Join-Path $Backend ".env.example"

function Test-PortOpen([int] $Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $result = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $opened = $result.AsyncWaitHandle.WaitOne(250, $false) -and $client.Connected
        return [bool] $opened
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

Write-Host "OperateSolve" -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The Python launcher 'py' was not found. Install Python 3.11+ and reopen PowerShell."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found. Install Node.js 20+ and reopen PowerShell."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating the Python virtual environment..."
    py -3 -m venv (Join-Path $Backend ".venv")
}

Write-Host "Checking Python dependencies..."
& $Python -m pip install -q -r (Join-Path $Backend "requirements.txt")

if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $Frontend
    try {
        npm install
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path $EnvFile)) {
    Copy-Item $EnvExample $EnvFile
    Write-Warning "Created backend\.env from the example. Add OPENAI_API_KEY before chatting."
}

$configuredKey = -not [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)
foreach ($line in Get-Content -LiteralPath $EnvFile) {
    if ($line -match '^\s*OPENAI_API_KEY\s*=\s*(.+)\s*$') {
        $value = $Matches[1].Trim().Trim('"').Trim("'")
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $configuredKey = $true
        }
    }
}
if (-not $configuredKey) {
    Write-Warning "OPENAI_API_KEY is empty in backend\.env. The UI will load, and model calls will fail until you add a key and restart the API window."
}

if (Test-PortOpen 8000) {
    Write-Host "API already listening on http://127.0.0.1:8000"
} else {
    Start-Process -FilePath "powershell.exe" -WorkingDirectory $Backend -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000"
    ) | Out-Null
    Write-Host "API window started:  http://127.0.0.1:8000/api/health"
}

if (Test-PortOpen 3000) {
    Write-Host "App already listening on http://127.0.0.1:3000"
} else {
    Start-Process -FilePath "powershell.exe" -WorkingDirectory $Frontend -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "npm run dev -- --hostname 127.0.0.1 --port 3000"
    ) | Out-Null
    Write-Host "App window started:  http://127.0.0.1:3000"
}

Write-Host ""
Write-Host "Browser API path: /api/* (Next.js proxies to 127.0.0.1:8000)."
Write-Host "Close those two windows to stop the app."
