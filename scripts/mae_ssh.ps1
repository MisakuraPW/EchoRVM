param([string]$Command, [switch]$Close, [string]$PipeName = 'codex_mae_monitor')
$ErrorActionPreference = 'Stop'
if (-not $Close -and -not $Command) { throw 'A command or -Close is required' }
$pipe = [IO.Pipes.NamedPipeClientStream]::new('.',$PipeName,[IO.Pipes.PipeDirection]::InOut)
try {
    try { $pipe.Connect(2000) } catch [TimeoutException] {
        if ($Close) { throw 'No existing MAE SSH channel' }
        $helper = Join-Path $PSScriptRoot 'mae_ssh_server.ps1'
        $shell = (Get-Process -Id $PID).Path
        Start-Process -FilePath $shell -WindowStyle Hidden -ArgumentList @('-NoProfile','-File',('"'+$helper+'"'),'-PipeName',$PipeName) | Out-Null
        $pipe.Connect(20000)
    }
    $reader = [IO.StreamReader]::new($pipe,[Text.UTF8Encoding]::new($false),$false,4096,$true)
    $writer = [IO.StreamWriter]::new($pipe,[Text.UTF8Encoding]::new($false),4096,$true)
    $writer.AutoFlush = $true
    $writer.WriteLine((@{command=$Command;close=[bool]$Close} | ConvertTo-Json -Compress))
    $pending = $reader.ReadLineAsync()
    if (-not $pending.Wait(150000)) { throw 'SSH channel response timeout; do not blindly repeat a remote action' }
    $response = $pending.GetAwaiter().GetResult() | ConvertFrom-Json
    if ($null -eq $response) { throw 'SSH channel failed without a response' }
    $response | ConvertTo-Json -Compress
    $writer.Dispose()
    $reader.Dispose()
    exit ([int]$response.exit_code)
} finally { $pipe.Dispose() }
