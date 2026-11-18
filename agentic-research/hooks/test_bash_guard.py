#!/usr/bin/env python3
"""Pipe-tests of bash_guard.py: each case is (command, expected decision). Run after any edit of the guard:

    python .claude/hooks/test_bash_guard.py

The cases live here, not on a command line, because the guard scans the whole Bash command and would
deny the test itself.
"""
import json
import subprocess
import sys
from pathlib import Path

GUARD = Path(__file__).with_name("bash_guard.py")

CASES = [
    # a note or a script written through a heredoc is data, not a command (false positive of 11-15)
    ("cat >> notes.md <<'EOF'\nidea: `git init` + first commit; never scancel -u; python submit_coral.py\nEOF\necho done", "allow"),
    ("python - <<'EOF'\nprint('git push upstream x')\nEOF\necho ok", "allow"),
    ("ssh host bash <<'EOF'\nscancel -u bcarew\nEOF", "deny"),          # a heredoc fed to a shell is a command
    # commit messages
    ("git commit -q -m 'Remove the upstream refit'", "allow"),
    ("git commit -m \"$(cat <<'EOF'\nSubject\n\nbody\nEOF\n)\"", "deny"),
    ("git commit -F - <<EOF\nSubject\nEOF", "deny"),
    ("git add a && git commit -q -m 'Fix the reader' && git commit -m 'Second'", "deny"),
    ("git commit -m 'Fix' -m 'body'", "deny"),
    ("git commit -m 'Fix the reader\n\nCo-Authored-By: x'", "deny"),
    ("git commit -m 'R75 near-wall seeds'", "deny"),
    ("git commit -m '" + "x" * 61 + "'", "deny"),
    # upstream, slot main, cluster
    ("git push upstream ben/dev", "deny"),
    ("git push -q origin ben/dev", "allow"),
    ("python - <<'EOF'\nprint('x')\nEOF\ngit -C Software/Coral checkout main", "deny"),
    ("cd Software/worktrees/a && git checkout -b ben/x", "allow"),
    ("scancel -u bcarew", "deny"),
    ("python exp.py cancel --all", "deny"),
    ("python /sps/hyperk/bcarew/merge_coral.py --run x", "deny"),
    # reminder only
    ("sextant run Analysis/Experiments/x.ini --sync", "remind"),
    ("ls", "allow"),
]


def decision(command):
    payload = json.dumps({"tool_input": {"command": command}, "cwd": "/x"})
    out = subprocess.run([sys.executable, str(GUARD)], input=payload, capture_output=True, text=True).stdout
    if not out.strip():
        return "allow"
    hso = json.loads(out)["hookSpecificOutput"]
    return hso.get("permissionDecision") or ("remind" if hso.get("additionalContext") else "allow")


def main():
    bad = 0
    for command, want in CASES:
        got = decision(command)
        ok = got == want
        bad += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {want:6s} {got:6s}  {command.splitlines()[0][:70]}")
    print(f"{len(CASES) - bad} / {len(CASES)} pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
