# Project delivery workflow

- The user has authorized automatic commits and pushes for completed code changes in this project. After implementing and verifying a code change, commit the task-scoped files and push the current branch to its configured upstream without requiring another reminder.
- Inspect the working tree and index first. Preserve unrelated edits and staged work; never include them implicitly in a task commit.
- Do not commit credentials, datasets, model checkpoints, generated outputs, or unrelated research records as part of code delivery.
- Never force-push or overwrite remote changes. If verification, authentication, networking, or remote divergence blocks delivery, report the blocker and do not claim the server can pull the change yet.
- Report the pushed commit and the server-side pull/run commands when relevant. Do not stop or restart server workloads unless separately authorized.
