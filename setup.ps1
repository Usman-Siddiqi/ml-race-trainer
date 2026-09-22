param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/'
}

# Keep Python/CUDA packages out of OneDrive and the source repository.
$raceEnvironment = Join-Path $env:LOCALAPPDATA 'RaceTrainer\venv'
$previousEnvironment = $env:UV_PROJECT_ENVIRONMENT
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:UV_PROJECT_ENVIRONMENT = $raceEnvironment
    & uv sync --locked --python 3.12 --extra $Device --group dev
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed with exit code $LASTEXITCODE"
    }
    Write-Output "Ready: $raceEnvironment"
    Write-Output 'Open Launch Race Lab.cmd. Setup does not run training or GPU workloads.'
}
finally {
    Pop-Location
    if ($null -eq $previousEnvironment) {
        Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
    }
    else {
        $env:UV_PROJECT_ENVIRONMENT = $previousEnvironment
    }
}
