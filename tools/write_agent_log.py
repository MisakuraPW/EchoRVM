"""Create a local, git-ignored Codex change log entry."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a local agent log markdown file.")
    parser.add_argument("--title", default="codex_update")
    parser.add_argument("--body", default="")
    args = parser.parse_args()

    safe_title = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.title).strip("_")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = Path("agent_logs")
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{timestamp}_{safe_title or 'codex_update'}.md"
    body = args.body or "请在这里补充本轮 Codex 的修改摘要、验证命令和下一步。"
    path.write_text(f"# {args.title}\n\n{body}\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
