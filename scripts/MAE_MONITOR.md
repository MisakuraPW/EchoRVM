# MAE Read-Only Monitor

The local Windows entry point is `scripts/Get-MaeMonitor.ps1` (PowerShell 7).
It sends the stdlib-only snapshot reader over an existing SSH channel without
uploading files, importing the training package, or changing the remote checkout.

`mae_ssh.ps1` talks to a same-Windows-user-only named pipe. If absent, it starts
`mae_ssh_server.ps1` in a hidden local process. The helper holds one authenticated
OpenSSH connection to the existing `autodl-mae` alias, with keepalives. It closes
after 30 idle minutes. It uses BatchMode/key authentication and strict host-key
checking: no passwords, askpass, credentials in scripts, network listener, or
SSH config edits. A lost connection fails visibly; a later call can reconnect
with the key. Do not blindly retry state-changing commands.

```powershell
& ./scripts/Get-MaeMonitor.ps1
# Optional explicit close, not needed during active monitoring:
& ./scripts/mae_ssh.ps1 -Close
```

The result is an outer JSON response (`exit_code`, `output`), with a JSON snapshot
inside `output`. It includes project-scoped processes, queue events, current stage
heartbeat/log tail, recent epochs, GPU and disk. It does not load model weights
or dataset pixels. Defaults are the currently authorized `cr_forward_v1` run,
its smoke run, `/root/autodl-tmp/EchoRVM`, and `outputs_cr_clean` on AutoDL.

The app automation `mae` uses GPT-6 Luna / Low every ten minutes. It only reads
this snapshot and updates ignored `outputs/monitor/cr_forward_v1/watch_state.json`.
New actionable anomalies go to the authorized main chat; healthy checks remain
silent and each independent automation chat calls the archive tool. The prompt
explicitly forbids archiving the main chat, changing training, restarting,
pulling code, touching other projects, or treating a budget pause as failure.
Repeated alerts are deduplicated. Completed runs are confirmed twice before
the automation pauses itself.

This is a local app automation, not a daemon on AutoDL. The Windows computer
and Codex app must remain running; SSH access and tool permissions must remain
available. Creation confirms configuration, not that all future runs are
guaranteed to succeed. The initial manual snapshot and channel reuse were tested.
