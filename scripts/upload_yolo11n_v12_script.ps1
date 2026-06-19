param(
    [Parameter(Mandatory = $true)]
    [string]$Remote,

    [int]$Port = 22,
    [string]$RemoteDir = "AI-Data"
)

$ErrorActionPreference = "Stop"

function Quote-BashSingle {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$LocalTrainScript = Join-Path $RootDir "scripts\train_yolo11n_gpu.sh"

if (-not (Test-Path $LocalTrainScript)) {
    throw "Missing local train script: $LocalTrainScript"
}

$remoteDirQ = Quote-BashSingle $RemoteDir

ssh -p $Port $Remote "mkdir -p $remoteDirQ/scripts"
scp -P $Port $LocalTrainScript "${Remote}:$RemoteDir/scripts/train_yolo11n_gpu.sh"
ssh -p $Port $Remote "chmod +x $remoteDirQ/scripts/train_yolo11n_gpu.sh && ls -l $remoteDirQ/scripts/train_yolo11n_gpu.sh"
