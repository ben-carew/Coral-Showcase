#!/usr/bin/env python3
"""PreToolUse(Bash) guard for the hard rules in CLAUDE.md. Reads the hook JSON on stdin.

Denies: commit messages with a body / attribution / long subject; pushes to upstream (git.hyperk.org);
git checkout|switch in Software/Coral (slot main); scancel -u / exp.py cancel --all; running the cluster
submit_*/merge_* scripts directly. Reminds (no block) before exp.py submit / sextant run.
"""
import json
import re
import shlex
import sys

SUBJECT_MAX = 60


def out(decision=None, reason=None, context=None):
    hso = {"hookEventName": "PreToolUse"}
    if decision:
        hso["permissionDecision"] = decision
        hso["permissionDecisionReason"] = reason
    if context:
        hso["additionalContext"] = context
    print(json.dumps({"hookSpecificOutput": hso}))
    sys.exit(0)


def deny(reason):
    out("deny", "CLAUDE.md hard rule: " + reason)


def commit_messages(cmd):
    """Every -m / --message value of a git commit in cmd, or None if they cannot be parsed."""
    try:
        tok = shlex.split(cmd, posix=True)
    except ValueError:
        return None
    msgs, in_commit = [], False
    i = 0
    while i < len(tok):
        t = tok[i]
        if t in ("&&", "||", ";", "|"):
            in_commit = False
        elif t == "commit" and i and any(x == "git" or x.endswith("/git") for x in tok[max(0, i - 6):i]):
            in_commit = True
        elif in_commit:
            if t in ("-m", "--message") and i + 1 < len(tok):
                msgs.append(tok[i + 1]); i += 1
            elif t.startswith("--message="):
                msgs.append(t.split("=", 1)[1])
            elif re.fullmatch(r"-[a-zA-Z]*m", t) and i + 1 < len(tok):   # -am "msg"
                msgs.append(tok[i + 1]); i += 1
        i += 1
    return msgs


HEREDOC = re.compile(r"(<<-?\s*(['\"]?)(\w+)\2[^\n]*\n).*?\n[ \t]*\3[ \t]*(?=\n|$)", re.S)


def without_heredoc_bodies(cmd):
    """The command with every here-document's BODY removed (the `<<TAG` opener stays).

    The rules below are about what the shell runs. A note or a script written through a heredoc is data:
    a notes line about a first repository snapshot was denied as a heredoc message (11-15)."""
    def body_is_data(m):
        line = cmd[cmd.rfind("\n", 0, m.start()) + 1:m.start()]
        # a heredoc fed to a shell (or to ssh) IS a command: keep it under the rules
        return not re.search(r"\b(bash|zsh|sh|ssh)\b", line)
    return HEREDOC.sub(lambda m: m.group(1) if body_is_data(m) else m.group(0), cmd)


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    cmd = (data.get("tool_input") or {}).get("command") or ""
    cwd = data.get("cwd") or ""
    if not cmd:
        sys.exit(0)
    cmd = without_heredoc_bodies(cmd)

    # 1. never push to upstream
    if re.search(r"\bgit\b[^|;&\n]*\bpush\b[^|;&\n]*(\bupstream\b|git\.hyperk\.org)", cmd):
        deny("never push to upstream (git.hyperk.org). Prepare the branch and hand bcarew the exact command.")

    # 2. commit messages: one short subject line, nothing else
    if re.search(r"\bgit\b[^|;&\n]*\bcommit\b", cmd):
        if re.search(r"co-authored-by|generated with|noreply@anthropic|claude code|\U0001F916", cmd, re.I):
            deny("no Co-Authored-By or any attribution line in commit messages.")
        if re.search(r"\bcommit\b[^|;&\n]*(\s-F\b|--file\b)", cmd):
            deny("commit with -m and ONE short subject line (no message file).")
        msgs = commit_messages(cmd)
        if msgs is None or (not msgs and "<<" in cmd):
            deny("commit with a plain -m 'Short imperative subject' (no heredoc, one line, no body).")
        if len(msgs) > 1:
            deny("one -m only: a commit message is one short subject line, no body.")
        for m in msgs:
            lines = [l for l in m.strip().splitlines() if l.strip()]
            if len(lines) != 1:
                deny("the commit message must be exactly one short subject line (no body).")
            if len(lines[0]) > SUBJECT_MAX:
                deny(f"commit subject is {len(lines[0])} chars; keep it to a few words (<= {SUBJECT_MAX}).")
            if re.search(r"\bR\d{1,3}[a-z]?\b|\br\d{2}[a-z0-9_]*_(e|mu|bench|val|smoke)\b|sextant", lines[0], re.I):
                deny("no roadmap ids, run names or 'Sextant' in commit subjects (internal jargon).")

    # 3. slot main (Software/Coral) is bcarew's: never checkout / switch there
    cur = cwd
    for seg in re.split(r"&&|\|\||;|\n", cmd):
        seg = seg.strip()
        m = re.match(r"cd\s+(\S+)", seg)
        if m:
            cur = m.group(1).strip("'\"")
            continue
        if re.search(r"\bgit\b.*\b(checkout|switch)\b", seg):
            c = re.search(r"-C\s+(\S+)", seg)
            where = c.group(1).strip("'\"") if c else cur
            if re.search(r"Software/Coral/?$", where.rstrip("/") + "/") or re.search(r"Software/Coral(/|$)", where):
                deny("never git checkout / switch in Software/Coral (slot main is bcarew's). Use Software/worktrees/{a,b,c} (wt.py list).")

    # 4. never cancel every job; never drive the cluster scripts directly
    if re.search(r"\bscancel\b[^|;&\n]*\s-u\b|exp\.py\s+cancel\s+--all", cmd):
        deny("never scancel -u / exp.py cancel --all (kills other sessions' jobs). Cancel one run: exp.py cancel <run>.")
    if re.search(r"(python[\d.]*\s+|\./|zsh\s+|bash\s+)\S*(submit|merge)_(coral|fitqun)\.py", cmd):
        deny("exp.py is the ONLY submit/merge entry point; never run submit_*/merge_* directly.")

    # 5. reminder before a submission (does not block)
    if "<<" not in cmd and re.search(r"exp\.py\s+submit\b|\bsextant\s+run\b|\bsbatch\b", cmd):
        out(context=(
            "Submission checklist (CLAUDE.md / experiment skill): (1) ledger grepped for this idea, "
            "(2) pass/kill criterion already written in items/Rxx.md, (3) code change -> 2-arm x 4-event smoke "
            "first and the arms differ where they should, (4) <= 1000 jobs, held-out files, (5) verify alive "
            "within ~3 min, (6) give bcarew the paste-ready monitor command."))
    sys.exit(0)


if __name__ == "__main__":
    main()
