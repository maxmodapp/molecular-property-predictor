param(
    [string]$OutputRoot = "",
    [switch]$CreateZip,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not $OutputRoot) {
    $OutputRoot = $projectRoot
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)

Write-Host "Proyecto:" $projectRoot
Write-Host "Salida:" $OutputRoot

$pyinstallerCheck = python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('PyInstaller') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "PyInstaller no esta instalado." -ForegroundColor Yellow
    Write-Host "Instalalo con:" -ForegroundColor Yellow
    Write-Host "    python -m pip install pyinstaller"
    exit 1
}

Write-Host ""
Write-Host "Generando ejecutable..." -ForegroundColor Cyan

$buildRoot = Join-Path $OutputRoot "build"
$specBuildRoot = Join-Path $buildRoot "desktop_app_final"
$distRoot = Join-Path $OutputRoot "dist"
$portableDir = Join-Path $distRoot "PrediccionPropiedadesQuimicas"
$portableZip = Join-Path $distRoot "PrediccionPropiedadesQuimicas_portable.zip"

New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $specBuildRoot | Out-Null
New-Item -ItemType Directory -Force -Path $distRoot | Out-Null

python -m py_compile desktop_app_final.py

$pyinstallerArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--workpath",
    $buildRoot,
    "--distpath",
    $distRoot,
    "desktop_app_final.spec"
)
if ($Clean) {
    $pyinstallerArgs = @("-m", "PyInstaller", "--clean") + $pyinstallerArgs[2..($pyinstallerArgs.Count - 1)]
}

python @pyinstallerArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "La generacion del ejecutable fallo." -ForegroundColor Red
    exit $LASTEXITCODE
}

if ($CreateZip) {
    if (Test-Path $portableZip) {
        Remove-Item -Force $portableZip
    }
    Compress-Archive -Path $portableDir -DestinationPath $portableZip -Force
}

Write-Host ""
Write-Host "Listo." -ForegroundColor Green
Write-Host "Ejecutable:" (Join-Path $portableDir "PrediccionPropiedadesQuimicas.exe")
if ($CreateZip) {
    Write-Host "ZIP portable:" $portableZip
} else {
    Write-Host "Carpeta portable:" $portableDir
}
