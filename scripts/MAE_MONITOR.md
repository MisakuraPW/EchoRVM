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
& ./scripts/Get-MaeMonitor.ps1 -RunTag cr_forward_fast_v3
# Optional explicit close, not needed during active monitoring:
& ./scripts/mae_ssh.ps1 -Close
```

The result is an outer JSON response (`exit_code`, `output`), with a JSON snapshot
inside `output`. It includes project-scoped processes, queue events, current stage
heartbeat/log tail, recent epochs, GPU and disk. It does not load model weights
or dataset pixels. Defaults are the currently authorized `cr_forward_fast_v3` run,
its smoke run, `/root/autodl-tmp/EchoRVM`, and `outputs_cr_clean` on AutoDL.

The app cron automation `mae` is PAUSED as of 2026-10-29. Its two visible run chats
were explicitly archived. One scheduled run performed no inspection; another
could not access the local named pipe, and neither actually called the archive
tool. A prompt promising automatic archiving is not a verified implementation.
Do not reactivate this cron configuration or claim it is monitoring training.
Manual snapshots remain available. An existing-thread heartbeat avoids separate
run chats, but does not provide this cron's per-run cheap-model override; that
tradeoff must be agreed before replacing the monitor.

This is a local app automation, not a daemon on AutoDL. The Windows computer
and Codex app must remain running; SSH access and tool permissions must remain
available. Creation confirms configuration, not that all future runs are
guaranteed to succeed. The initial manual snapshot and channel reuse were tested.

## Live terminal progress

Future queue launches display a stage counter and a read-only status line every
ten seconds (epoch/sample progress, loss, throughput, epoch ETA, memory, data wait,
heartbeat and last validation metrics). Full stdout remains in `worker.log`;
worker failures also print the recent log tail to the launching terminal.

An already running parent cannot import this display retroactively. Use the
standalone stdlib viewer in a second SSH terminal instead. It only reads output
files and can live outside the checkout to avoid changing a locked code hash:

```bash
python tools/watch_pipeline_progress.py --tag cr_forward_fast_v3
```

Ctrl+C closes this separate viewer only. It does not signal the training worker.
Heartbeat frequency is fifteen seconds; polling faster does not invent per-step
updates. ETA is approximate for the current epoch, not the whole multi-stage run.
On older runs without the selection sidecar, the viewer intentionally omits a
total stage count instead of guessing from disabled/future stages in the plan.
