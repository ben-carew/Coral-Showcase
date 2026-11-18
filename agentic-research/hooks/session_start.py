#!/usr/bin/env python3
"""SessionStart hook: a ten-line orientation so a session does not spend tool calls finding out where it is.
Local only (no ssh), silent on any error. Prints: laptop slots (branch, sha, dirty), the top of the roadmap queue,
the newest ledger dates. Wired in .claude/settings.json."""
import re
import subprocess
from pathlib import Path

ROOT = Path("/Users/bcarew/Documents/PhD/Reconstruction")
SLOTS = {"main": ROOT / "Software/Coral", "a": ROOT / "Software/worktrees/a",
         "b": ROOT / "Software/worktrees/b", "c": ROOT / "Software/worktrees/c"}


def git(path, *args):
    out = subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, timeout=3)
    return out.stdout.strip()


def main():
    lines = ["[session start] laptop slots (server slots: Analysis/Scripts/wt.py list):"]
    for name, path in SLOTS.items():
        if not path.is_dir():
            continue
        branch = git(path, "rev-parse", "--abbrev-ref", "HEAD")
        sha = git(path, "rev-parse", "--short", "HEAD")
        dirty = "dirty" if git(path, "status", "--porcelain", "--untracked-files=no") else "clean"
        note = "  <- bcarew's, never checkout here" if name == "main" else ""
        lines.append(f"  {name:4s} {branch:28s} {sha:8s} {dirty}{note}")
    notes = ROOT / ".claude/Project Notes"
    roadmap = (notes / "roadmap.md").read_text()
    queue = re.findall(r"^\*\*(\d+ · [^*]+)\*\*", roadmap, flags=re.M)
    if queue:
        lines.append("roadmap queue, top 4: " + "; ".join(q.strip()[:70] for q in queue[:4]))
    dates = re.findall(r"^\| (\d\d-\d\d) \|", (notes / "ledger.md").read_text(), flags=re.M)
    if dates:
        lines.append(f"ledger: {len(dates)} rows, newest {max(dates)}. grep it before proposing or testing anything.")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
