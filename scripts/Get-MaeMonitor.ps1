$ErrorActionPreference = 'Stop'
$program = Get-Content -LiteralPath (Join-Path $PSScriptRoot '../tools/pipeline_monitor_snapshot.py') -Raw
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($program))
$command = "/root/miniconda3/envs/echocardmae/bin/python -c `"import base64;exec(base64.b64decode('$encoded'))`""
& (Join-Path $PSScriptRoot 'mae_ssh.ps1') -Command $command
exit $LASTEXITCODE
