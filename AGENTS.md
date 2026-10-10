# Project delivery workflow

- Use a single agent for this project by default. Do not spawn subagents or introduce multi-agent implementation/review unless the user explicitly requests it.
- The user has authorized automatic commits and pushes for completed code changes in this project. After implementing and verifying a code change, commit the task-scoped files and push the current branch to its configured upstream without requiring another reminder.
- Inspect the working tree and index first. Preserve unrelated edits and staged work; never include them implicitly in a task commit.
- Do not commit credentials, datasets, model checkpoints, generated outputs, or unrelated research records as part of code delivery.
- Never force-push or overwrite remote changes. If verification, authentication, networking, or remote divergence blocks delivery, report the blocker and do not claim the server can pull the change yet.
- Report the pushed commit and the server-side pull/run commands when relevant. Do not stop or restart server workloads unless separately authorized.
- The authoritative research workbook and its maintenance documents also belong to this repository (`MisakuraPW/EchoRVM`). After an authorized research-record update and verification, commit and push those task-scoped files automatically as well. Separate commits are allowed, but do not leave a completed authorized workbook update unpublished merely because it is not code. Follow `docs/AGENTS.md` for content permissions.
