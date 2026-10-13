# Starts the API and the web studio, then opens the studio in the browser.
# Ctrl+C stops only the processes this script started.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$ApiUrl = "http://127.0.0.1:8080/docs"
$StudioUrl = "http://127.0.0.1:5173/"

function Test-Listening([int] $Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $client.Connect("127.0.0.1", $Port)
        return $true
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-Url([string] $Url, [int] $Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 400
        }
    }
    return $false
}

function Stop-ProcessTree($Process) {
    if (-not $Process) { return }
    try {
        if (-not $Process.HasExited) {
            & taskkill.exe /PID $Process.Id /T /F | Out-Null
        }
    } catch {
        # The process already exited.
    }
}

$ffmpeg = Join-Path $Root ".tools\ffmpeg\bin"
if (Test-Path (Join-Path $ffmpeg "ffmpeg.exe")) {
    $env:PATH = "$ffmpeg;$env:PATH"
}

$python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "The project Python environment is missing: $python"
    Write-Host "Create .venv and install requirements, then run this script again."
    exit 1
}

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    Write-Host "npm was not found. Install Node.js, then run this script again."
    exit 1
}

$frontend = Join-Path $Root "frontend"
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing web app dependencies..."
    Push-Location $frontend
    try {
        & npm.cmd install
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Pop-Location
    }
}

$api = $null
$web = $null
$startedApi = -not (Test-Listening 8080)
$startedWeb = -not (Test-Listening 5173)
$logDir = Join-Path $Root "storage\run"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

try {
    if ($startedApi) {
        Write-Host "Starting the API..."
        $api = Start-Process -FilePath $python -ArgumentList "main.py" -WorkingDirectory $Root -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "api.log") `
            -RedirectStandardError (Join-Path $logDir "api.err.log")
    } else {
        Write-Host "The API is already running."
    }

    if ($startedWeb) {
        Write-Host "Starting the studio..."
        $web = Start-Process -FilePath "cmd.exe" -ArgumentList "/d","/c","npm run dev" -WorkingDirectory $frontend -PassThru -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $logDir "studio.log") `
            -RedirectStandardError (Join-Path $logDir "studio.err.log")
    } else {
        Write-Host "The studio is already running."
    }

    if (-not (Wait-Url $ApiUrl 45)) {
        throw "The API did not answer at $ApiUrl. See storage\run\api.err.log"
    }
    if (-not (Wait-Url $StudioUrl 45)) {
        throw "The studio did not answer at $StudioUrl. See storage\run\studio.err.log"
    }

    Write-Host ""
    Write-Host "Studio  $StudioUrl"
    Write-Host "API     $ApiUrl"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop what this script started."
    Start-Process $StudioUrl | Out-Null

    while ($true) {
        Start-Sleep -Seconds 2
        if ($api -and $api.HasExited) { throw "The API stopped." }
        if ($web -and $web.HasExited) { throw "The studio stopped." }
    }
} finally {
    if ($startedApi) { Stop-ProcessTree $api }
    if ($startedWeb) { Stop-ProcessTree $web }
}
