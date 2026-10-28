param([string]$PipeName = 'codex_mae_monitor')
$ErrorActionPreference = 'Stop'
$start = [Diagnostics.ProcessStartInfo]::new()
$start.FileName = 'C:\Windows\System32\OpenSSH\ssh.exe'
$start.UseShellExecute = $false
$start.CreateNoWindow = $true
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
foreach ($argument in @('-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
                        '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','autodl-mae','bash --noprofile --norc')) {
    $start.ArgumentList.Add($argument)
}
$ssh = [Diagnostics.Process]::Start($start)
$errors = $ssh.StandardError.ReadToEndAsync()
try {
    $stop = $false
    while (-not $stop) {
        $options = [IO.Pipes.PipeOptions]::Asynchronous -bor [IO.Pipes.PipeOptions]::CurrentUserOnly
        $pipe = [IO.Pipes.NamedPipeServerStream]::new($PipeName,[IO.Pipes.PipeDirection]::InOut,1,
                 [IO.Pipes.PipeTransmissionMode]::Byte,$options)
        try {
            if (-not $pipe.WaitForConnectionAsync().Wait(1800000)) { break }
            $reader = [IO.StreamReader]::new($pipe,[Text.UTF8Encoding]::new($false),$false,4096,$true)
            $writer = [IO.StreamWriter]::new($pipe,[Text.UTF8Encoding]::new($false),4096,$true)
            $writer.AutoFlush = $true
            try {
                $request = $reader.ReadLine() | ConvertFrom-Json
                if ($request.close) {
                    $stop = $true
                    $response = @{exit_code=0; output='SSH channel closed'}
                } else {
                    if ($ssh.HasExited) { throw "SSH exited: $($errors.GetAwaiter().GetResult())" }
                    $marker = '__MAE_END_' + [guid]::NewGuid().ToString('N') + ':'
                    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($request.command))
                    $ssh.StandardInput.WriteLine("printf '%s' '$encoded' | base64 -d | bash 2>&1; status=`${PIPESTATUS[2]}; printf '\n$marker%s\n' `"`$status`"")
                    $ssh.StandardInput.Flush()
                    $output = [Text.StringBuilder]::new()
                    while ($true) {
                        $line = $ssh.StandardOutput.ReadLineAsync()
                        if (-not $line.Wait(120000)) { throw 'Read-only command timeout; inspect before retrying' }
                        $value = $line.GetAwaiter().GetResult()
                        if ($null -eq $value) { throw "SSH disconnected: $($errors.GetAwaiter().GetResult())" }
                        if ($value.StartsWith($marker)) {
                            $response = @{exit_code=[int]$value.Substring($marker.Length); output=$output.ToString()}
                            break
                        }
                        [void]$output.AppendLine($value)
                    }
                }
            } catch {
                $response = @{exit_code=255; output=$_.Exception.Message}
                $stop = $true
            }
            $writer.WriteLine(($response | ConvertTo-Json -Compress))
            $writer.Dispose()
            $reader.Dispose()
        } finally { $pipe.Dispose() }
    }
} finally {
    if (-not $ssh.HasExited) {
        $ssh.StandardInput.Close()
        if (-not $ssh.WaitForExit(2000)) { $ssh.Kill() }
    }
    $ssh.Dispose()
}
